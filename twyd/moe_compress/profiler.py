"""Calibration-time functional profiling for MoE layers.

Computes for a single MoE layer on N calibration tokens X (N, H):
  - token_counts[E]: activation count per expert
  - gate_mass[E]: sum of active router weights per expert
  - output_energy[E]: sum_n ||g_{n,e} f_e(x_n)||^2
  - gram[E, E] (fp64): exact functional Gram matrix G_{e,e'} = <Z_e, Z_{e'}>_F
    where Z_e[n] = g_{n,e} f_e(x_n) if e in Top-m(x_n) else 0
  - func_sim[E, E] (fp32): permutation-invariant functional cosine similarity
    on a shared stratified probe batch X_probe (P, H)
  - teacher_y[N, H]: exact teacher MoE output Y^(T)
  - router_probs[N, E]: full teacher softmax probabilities
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import MoEAdapter


@dataclass
class LayerProfile:
    layer_idx: int
    num_experts: int
    top_k: int
    norm_topk_prob: bool
    num_tokens: int
    token_counts: torch.Tensor     # [E] int64 (CPU)
    gate_mass: torch.Tensor        # [E] float64 (CPU)
    output_energy: torch.Tensor    # [E] float64 (CPU)
    gram: torch.Tensor             # [E, E] float64 (CPU)
    func_sim: torch.Tensor         # [E, E] float32 (CPU)
    teacher_y: torch.Tensor        # [N, H] same dtype as X (CPU)
    router_probs: torch.Tensor     # [N, E] float16/bfloat16 (CPU)


@torch.no_grad()
def profile_moe_layer(
    adapter: MoEAdapter,
    moe_block: nn.Module,
    x_flat: torch.Tensor,
    layer_idx: int = 0,
    chunk_size: int = 512,
    probe_tokens: int = 512,
    device: torch.device | str | None = None,
) -> LayerProfile:
    """Profile one MoE layer on flattened calibration hidden states `x_flat` (N, H)."""
    if x_flat.ndim != 2:
        raise ValueError(f"Expected 2D x_flat (N, H), got {tuple(x_flat.shape)}")

    N, H = x_flat.shape
    E = adapter.get_num_experts(moe_block)
    top_k = adapter.get_top_k(moe_block)
    norm_topk = adapter.get_norm_topk_prob(moe_block)
    router_w = adapter.get_router_weight(moe_block)
    gate_up_proj, down_proj, act_fn = adapter.get_expert_weights(moe_block)

    dev = torch.device(device) if device is not None else router_w.device
    w_dtype = router_w.dtype
    router_w_d = router_w.to(device=dev)
    gate_up_d = gate_up_proj.to(device=dev)
    down_d = down_proj.to(device=dev)

    token_counts = torch.zeros(E, dtype=torch.int64, device=dev)
    gate_mass = torch.zeros(E, dtype=torch.float64, device=dev)
    gram = torch.zeros((E, E), dtype=torch.float64, device=dev)

    teacher_y = torch.empty((N, H), dtype=w_dtype, device="cpu")
    router_probs_out = torch.empty((N, E), dtype=torch.bfloat16, device="cpu")

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xb = x_flat[start:end].to(device=dev, dtype=w_dtype)
        B = end - start

        _, probs, top_w, top_idx = adapter.route_with_weight(
            xb, router_w_d, top_k, norm_topk
        )
        router_probs_out[start:end] = probs.to(dtype=torch.bfloat16, device="cpu")

        flat_idx = top_idx.reshape(-1)
        flat_w = top_w.reshape(-1).to(torch.float64)
        token_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int64))
        gate_mass.scatter_add_(0, flat_idx, flat_w)

        # Compute per-slot weighted expert outputs Z_slot: [B, top_k, H] in fp32
        z_slots = torch.zeros((B, top_k, H), dtype=torch.float32, device=dev)
        # Group by active expert for fast matmul
        unique_e = torch.unique(flat_idx)
        for e_t in unique_e:
            e = int(e_t.item())
            tok_pos, slot_pos = torch.where(top_idx == e)
            ye = adapter.expert_forward(xb[tok_pos], gate_up_d[e], down_d[e], act_fn)
            w_sel = top_w[tok_pos, slot_pos].unsqueeze(-1)
            z_slots[tok_pos, slot_pos] = (ye * w_sel).to(torch.float32)

        # Exact layer output = sum over top_k slots
        yb = z_slots.sum(dim=1)
        teacher_y[start:end] = yb.to(dtype=w_dtype, device="cpu")

        # Exact active functional Gram accumulation: G[e_a, e_b] += sum_b <z[b,a], z[b,b']>
        # Pairwise inner products per token: [B, top_k, top_k]
        pair_dots = torch.bmm(z_slots, z_slots.transpose(1, 2)).to(torch.float64)
        idx_a = top_idx.unsqueeze(2).expand(B, top_k, top_k).reshape(-1)
        idx_b = top_idx.unsqueeze(1).expand(B, top_k, top_k).reshape(-1)
        flat_pair = idx_a * E + idx_b
        gram.view(-1).scatter_add_(0, flat_pair, pair_dots.reshape(-1))

    # Symmetrize fp64 rounding noise
    gram = 0.5 * (gram + gram.T)
    output_energy = torch.diagonal(gram).clamp_min(0.0)

    # Permutation-invariant functional similarity on a stratified probe batch
    P = min(probe_tokens, N)
    if P == N:
        probe_idx = torch.arange(N, dtype=torch.long)
    else:
        probe_idx = torch.linspace(0, N - 1, steps=P).round().long()
    x_probe = x_flat[probe_idx].to(device=dev, dtype=w_dtype)

    # Compute each expert's output on x_probe, normalize to unit vector in R^(P*H)
    probe_vecs = torch.empty((E, P * H), dtype=torch.float32, device=dev)
    for e in range(E):
        ye = adapter.expert_forward(x_probe, gate_up_d[e], down_d[e], act_fn).to(torch.float32)
        probe_vecs[e] = ye.reshape(-1)
    norms = probe_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    probe_unit = probe_vecs / norms
    func_sim = (probe_unit @ probe_unit.T).clamp(-1.0, 1.0)
    func_sim.fill_diagonal_(1.0)

    return LayerProfile(
        layer_idx=layer_idx,
        num_experts=E,
        top_k=top_k,
        norm_topk_prob=norm_topk,
        num_tokens=N,
        token_counts=token_counts.cpu(),
        gate_mass=gate_mass.cpu(),
        output_energy=output_energy.cpu(),
        gram=gram.cpu(),
        func_sim=func_sim.cpu(),
        teacher_y=teacher_y,
        router_probs=router_probs_out,
    )
