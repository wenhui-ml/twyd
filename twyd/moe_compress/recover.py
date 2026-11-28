"""Stage 2 Online KD + Cluster-Router-KL Recovery for LRC-compressed native MoE models.

Designed for dual-GPU asymmetric execution on 2 x 24 GB GPUs:
  - Frozen MoE Teacher (e.g. OLMoE-1B-7B, 13.8 GB bf16) lives on `teacher_device` (cuda:1).
  - Trainable compressed native HF MoE Student (e.g. 32 or 16 experts, 4-7.4 GB bf16)
    lives on `student_device` (cuda:0).
  - Teacher extracts chunked Top-K logit targets + per-layer cluster-aggregated
    routing distributions on `teacher_device` and transfers only tiny target tensors
    to `student_device`.
  - Optional CPU-offloaded fp32 AdamW keeps optimizer states in host RAM (~1% step
    overhead at GAS=16..32, saving 24-44 GB VRAM with zero DeepSpeed/ROCm build risk).
  - Saves 100% native HF checkpoints (`student.save_pretrained`, zero trust_remote_code).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


@dataclass
class RecoveryConfig:
    kl_temperature: float = 1.0
    kl_topk: int = 256
    kl_weight: float = 2.0
    ce_weight: float = 1.0
    jsd_beta: float = 0.0                  # 0.0 = exact Forward KL; (0,1) = Generalized JSD
    router_kl_weight: float = 0.2
    router_target_mode: str = "survivor"   # "survivor" | "cluster"
    hidden_align_weight: float = 0.5       # same-width hidden cosine alignment weight
    hidden_align_layers: Tuple[int, ...] = (3, 7, 11, 15)
    aux_loss_weight: float = 0.01
    chunk_size: int = 512
    lr: float = 2e-4
    router_lr_scale: float = 0.5
    attn_lr_scale: float = 0.3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_rank: int = 64                    # >0 = mergeable per-expert LoRA (0 = full-rank)
    lora_alpha: float = 64.0
    trainable_scope: str = "moe"           # "moe" | "all_no_embed" | "all"
    cpu_offload_optimizer: bool = False


class MoeExpertLoRAWrapper(nn.Module):
    """Mergeable per-expert low-rank adapter wrapping native fused 3D `OlmoeExperts`.

    Forward adds `(A_gu @ B_gu)` to `gate_up_proj` and `(A_d @ B_d)` to `down_proj`
    per active expert without ever materializing full 3D dense updates on GPU, and
    folds 100% losslessly back into native `OlmoeExperts` tensors at export time.
    """

    def __init__(self, base_experts: nn.Module, rank: int = 64, alpha: float = 64.0):
        super().__init__()
        self.base_experts = base_experts
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)

        gu = base_experts.gate_up_proj  # [E, 2*d_ff, H]
        dn = base_experts.down_proj     # [E, H, d_ff]
        E, two_dff, H = gu.shape
        d_ff = dn.shape[2]
        dev, dt = gu.device, gu.dtype

        # Freeze base 3D weights
        base_experts.gate_up_proj.requires_grad_(False)
        base_experts.down_proj.requires_grad_(False)

        # gate_up delta: [E, 2*d_ff, H] = B_gu [E, 2*d_ff, r] @ A_gu [E, r, H]
        # Zero-init B so Step 0 is bit-exact with base_experts.
        self.gu_A = nn.Parameter(torch.empty((E, self.rank, H), device=dev, dtype=dt))
        self.gu_B = nn.Parameter(torch.zeros((E, two_dff, self.rank), device=dev, dtype=dt))
        self.dn_A = nn.Parameter(torch.empty((E, self.rank, d_ff), device=dev, dtype=dt))
        self.dn_B = nn.Parameter(torch.zeros((E, H, self.rank), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.gu_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dn_A, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_experts
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = int(expert_idx[0].item())
            if e == base.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            cur = hidden_states[token_idx]

            # gate_up: base + scale * (cur @ A^T) @ B^T
            gu_out = F.linear(cur, base.gate_up_proj[e]) + self.scale * F.linear(
                F.linear(cur, self.gu_A[e]), self.gu_B[e]
            )
            gate, up = gu_out.chunk(2, dim=-1)
            h = base.act_fn(gate) * up

            # down: base + scale * (h @ A^T) @ B^T
            dn_out = F.linear(h, base.down_proj[e]) + self.scale * F.linear(
                F.linear(h, self.dn_A[e]), self.dn_B[e]
            )
            dn_out = dn_out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, dn_out.to(final_hidden_states.dtype))

        return final_hidden_states

    @torch.no_grad()
    def merged_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact merged `(gate_up_proj, down_proj)` in base dtype."""
        gu_delta = torch.bmm(self.gu_B.float(), self.gu_A.float()) * self.scale
        dn_delta = torch.bmm(self.dn_B.float(), self.dn_A.float()) * self.scale
        new_gu = (self.base_experts.gate_up_proj.float() + gu_delta).to(self.base_experts.gate_up_proj.dtype)
        new_dn = (self.base_experts.down_proj.float() + dn_delta).to(self.base_experts.down_proj.dtype)
        return new_gu, new_dn


def inject_moe_expert_lora_(model: nn.Module, rank: int = 64, alpha: float = 64.0) -> int:
    """Wrap every MoE layer's `mlp.experts` with `MoeExpertLoRAWrapper` in-place."""
    if rank <= 0:
        return 0
    count = 0
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            if not isinstance(layer.mlp.experts, MoeExpertLoRAWrapper):
                layer.mlp.experts = MoeExpertLoRAWrapper(layer.mlp.experts, rank=rank, alpha=alpha)
                count += 1
    return count


@torch.no_grad()
def save_merged_native_moe(model: nn.Module, save_dir: str, tokenizer=None) -> None:
    """Save 100% native HF MoE checkpoint with all `MoeExpertLoRAWrapper` deltas merged in."""
    os.makedirs(save_dir, exist_ok=True)
    saved_wrappers: List[Tuple[nn.Module, MoeExpertLoRAWrapper, torch.Tensor, torch.Tensor]] = []
    try:
        for layer in model.model.layers:
            if hasattr(layer, "mlp") and isinstance(layer.mlp.experts, MoeExpertLoRAWrapper):
                wrap = layer.mlp.experts
                base = wrap.base_experts
                orig_gu = base.gate_up_proj.data.detach().cpu().clone()
                orig_dn = base.down_proj.data.detach().cpu().clone()
                merged_gu, merged_dn = wrap.merged_weights()
                base.gate_up_proj.data.copy_(merged_gu)
                base.down_proj.data.copy_(merged_dn)
                layer.mlp.experts = base
                saved_wrappers.append((layer.mlp, wrap, orig_gu, orig_dn))
        model.save_pretrained(save_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
    finally:
        # Restore training wrappers + unmerged base weights seamlessly
        for mlp, wrap, orig_gu, orig_dn in saved_wrappers:
            wrap.base_experts.gate_up_proj.data.copy_(orig_gu.to(wrap.base_experts.gate_up_proj.device))
            wrap.base_experts.down_proj.data.copy_(orig_dn.to(wrap.base_experts.down_proj.device))
            mlp.experts = wrap


def load_cluster_maps(
    cluster_map_path: str,
) -> Tuple[int, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Load `lrc_cluster_map.json` -> `(new_E, {layer: cluster_map[E]}, {layer: survivors[K']})`."""
    with open(cluster_map_path, "r") as f:
        obj = json.load(f)
    new_e = int(obj["new_experts"])
    cmaps: Dict[int, torch.Tensor] = {}
    survs: Dict[int, torch.Tensor] = {}
    for k, v in obj["layers"].items():
        cmaps[int(k)] = torch.tensor(v["cluster_map"], dtype=torch.long)
        survs[int(k)] = torch.tensor(v["survivors"], dtype=torch.long)
    return new_e, cmaps, survs


@torch.no_grad()
def extract_teacher_topk_targets(
    t_logits: torch.Tensor,
    top_k: int,
    T: float,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract chunked Top-K + tail-bucket teacher distribution without materializing full fp32 vocab."""
    shift_t = t_logits[:, :-1, :].contiguous()
    B, S, V = shift_t.shape
    N = B * S
    K = min(int(top_k), V)
    flat_t = shift_t.view(N, V)

    idx_chunks = []
    logp_chunks = []
    p_chunks = []
    tail_chunks = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tc = flat_t[start:end].float() / T
        log_z = torch.logsumexp(tc, dim=-1, keepdim=True)
        top_vals, top_idx = torch.topk(tc, k=K, dim=-1)
        logp_top = top_vals - log_z
        p_top = logp_top.exp()
        p_tail = (1.0 - p_top.sum(dim=-1)).clamp_min(1e-9)

        idx_chunks.append(top_idx.to(torch.int32))
        logp_chunks.append(logp_top.to(torch.bfloat16))
        p_chunks.append(p_top.to(torch.bfloat16))
        tail_chunks.append(p_tail.to(torch.bfloat16))

    return (
        torch.cat(idx_chunks, dim=0),
        torch.cat(logp_chunks, dim=0),
        torch.cat(p_chunks, dim=0),
        torch.cat(tail_chunks, dim=0),
    )


@torch.no_grad()
def aggregate_teacher_router_targets(
    teacher_router_logits: Sequence[torch.Tensor],
    cluster_maps: Dict[int, torch.Tensor],
    new_num_experts: int,
    survivors: Optional[Dict[int, torch.Tensor]] = None,
    mode: str = "survivor",
) -> List[torch.Tensor]:
    """Convert teacher per-layer router logits `[N, E]` into student prob targets `[N, K']`.

    - `mode="survivor"`: conditional softmax over the `K'` kept experts
      `softmax(z_t[:, survivors])`, preserving exact specialization without
      cluster-mass distortion under `norm_topk_prob=False`.
    - `mode="cluster"`: probability-mass-preserving cluster sum `sum_{e in C_j} p_t[e]`.
    """
    targets: List[torch.Tensor] = []
    for layer_idx, r_logits in enumerate(teacher_router_logits):
        if layer_idx not in cluster_maps:
            continue
        if mode == "survivor" and survivors is not None and layer_idx in survivors:
            s_idx = survivors[layer_idx].to(device=r_logits.device)
            p_c = F.softmax(r_logits.index_select(1, s_idx).float(), dim=-1)
        else:
            p_t = F.softmax(r_logits.float(), dim=-1)
            cmap = cluster_maps[layer_idx].to(device=p_t.device)
            p_c = torch.zeros((p_t.shape[0], new_num_experts), dtype=torch.float32, device=p_t.device)
            p_c.index_add_(1, cmap, p_t)
            p_c = p_c.clamp_min(1e-12)
            p_c = p_c / p_c.sum(dim=-1, keepdim=True)
        targets.append(p_c.to(torch.bfloat16))
    return targets


def _chunk_topk_kl_ce(
    s_c: torch.Tensor,
    top_idx_c: torch.Tensor,
    logp_t_top_c: torch.Tensor,
    p_t_top_c: torch.Tensor,
    p_t_tail_c: torch.Tensor,
    y_c: torch.Tensor,
    T: float,
    jsd_beta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Checkpointed per-chunk Top-K + Tail-Bucket Forward-KL or Generalized JSD + CE (fp32)."""
    s_f = s_c.float()
    logp_s_T = F.log_softmax(s_f / T, dim=-1)
    idx64 = top_idx_c.long()
    logp_s_top = torch.gather(logp_s_T, dim=-1, index=idx64)

    p_t_top = p_t_top_c.float()
    logp_t_top = logp_t_top_c.float()
    p_t_tail = p_t_tail_c.float()

    p_s_top = logp_s_top.exp()
    p_s_tail = (1.0 - p_s_top.sum(dim=-1)).clamp_min(1e-9)

    if jsd_beta <= 0.0:
        kl_top = (p_t_top * (logp_t_top - logp_s_top)).sum(dim=-1)
        kl_tail = p_t_tail * (p_t_tail.log() - p_s_tail.log())
        kl_tok = (kl_top + kl_tail) * (T * T)
    else:
        # Exact Generalized JSD_beta(P_T || P_S) over Top-K + Tail bucket
        b = float(jsd_beta)
        m_top = ((1.0 - b) * p_t_top + b * p_s_top).clamp_min(1e-12)
        m_tail = ((1.0 - b) * p_t_tail + b * p_s_tail).clamp_min(1e-12)
        log_m_top = m_top.log()
        log_m_tail = m_tail.log()
        kl_tm = (p_t_top * (logp_t_top - log_m_top)).sum(dim=-1) + p_t_tail * (p_t_tail.log() - log_m_tail)
        kl_sm = (p_s_top * (logp_s_top - log_m_top)).sum(dim=-1) + p_s_tail * (p_s_tail.log() - log_m_tail)
        # Scale by 1 / (b * (1 - b))? Keep canonical 2x JSD scale so magnitude matches KL.
        kl_tok = ((1.0 - b) * kl_tm + b * kl_sm) * (T * T) / max(b * (1.0 - b) * 4.0, 1e-6)

    if T == 1.0:
        ce_tok = F.nll_loss(logp_s_T, y_c.long(), ignore_index=-100, reduction="none")
    else:
        ce_tok = F.cross_entropy(s_f, y_c.long(), ignore_index=-100, reduction="none")
    return kl_tok, ce_tok


def compute_fused_logit_kl_ce(
    s_logits: torch.Tensor,
    labels: torch.Tensor,
    top_idx: torch.Tensor,
    logp_t_top: torch.Tensor,
    p_t_top: torch.Tensor,
    p_t_tail: torch.Tensor,
    T: float = 1.0,
    chunk_size: int = 512,
    use_checkpoint: bool = True,
    jsd_beta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute masked mean (kl_loss, ce_loss) over all valid next-token positions (`labels != -100`)."""
    shift_s = s_logits[:, :-1, :].contiguous()
    shift_y = labels[:, 1:].contiguous().view(-1)
    N, V = shift_s.shape[0] * shift_s.shape[1], shift_s.shape[2]
    flat_s = shift_s.view(N, V)

    valid = shift_y != -100
    kl_sum = torch.zeros((), dtype=torch.float32, device=s_logits.device)
    ce_sum = torch.zeros((), dtype=torch.float32, device=s_logits.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sc = flat_s[start:end]
        ic = top_idx[start:end]
        ltc = logp_t_top[start:end]
        ptc = p_t_top[start:end]
        ptail = p_t_tail[start:end]
        yc = shift_y[start:end]
        vc = valid[start:end]
        if not bool(vc.any().item()):
            continue

        if use_checkpoint and sc.requires_grad:
            kl_c, ce_c = _ckpt(
                _chunk_topk_kl_ce,
                sc, ic, ltc, ptc, ptail, yc, T, jsd_beta,
                use_reentrant=False,
            )
        else:
            kl_c, ce_c = _chunk_topk_kl_ce(sc, ic, ltc, ptc, ptail, yc, T, jsd_beta)

        vf = vc.float()
        kl_sum = kl_sum + (kl_c * vf).sum()
        ce_sum = ce_sum + (ce_c * vf).sum()

    denom = valid.float().sum().clamp_min(1.0)
    return kl_sum / denom, ce_sum / denom


@torch.no_grad()
def extract_teacher_hidden_targets(
    teacher_hidden_states: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
) -> List[torch.Tensor]:
    """Extract bf16 teacher hidden states after transformer blocks `layer_indices`.

    `teacher_hidden_states` has length `L + 1` (index 0 = embeddings, index `l + 1` = output of block `l`).
    """
    n_states = len(teacher_hidden_states)
    out: List[torch.Tensor] = []
    for l_idx in layer_indices:
        pos = int(l_idx) + 1
        if 0 <= pos < n_states:
            out.append(teacher_hidden_states[pos].detach().to(torch.bfloat16))
    return out


def compute_hidden_cosine_loss(
    student_hidden_states: Sequence[torch.Tensor],
    teacher_hidden_targets: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean `1 - cos(h_S, h_T)` across tapped layers (exact same-width alignment, zero extra params)."""
    if not student_hidden_states or not teacher_hidden_targets:
        dev = student_hidden_states[0].device if student_hidden_states else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=dev)

    dev = student_hidden_states[0].device
    mask_f = token_mask.reshape(-1).float().to(dev) if token_mask is not None else None
    denom = mask_f.sum().clamp_min(1.0) if mask_f is not None else None

    total = torch.zeros((), dtype=torch.float32, device=dev)
    count = 0
    n_states = len(student_hidden_states)
    t_iter = iter(teacher_hidden_targets)
    for l_idx in layer_indices:
        pos = int(l_idx) + 1
        if not (0 <= pos < n_states):
            continue
        ht = next(t_iter, None)
        if ht is None:
            break
        hs = student_hidden_states[pos].float().reshape(-1, student_hidden_states[pos].shape[-1])
        ht_d = ht.to(device=dev, dtype=torch.float32).reshape(-1, ht.shape[-1])
        cos = F.cosine_similarity(hs, ht_d, dim=-1)
        dist = 1.0 - cos
        if mask_f is not None:
            total = total + (dist * mask_f).sum() / denom
        else:
            total = total + dist.mean()
        count += 1
    return total / max(count, 1)


def compute_router_kl_loss(
    student_router_logits: Sequence[torch.Tensor],
    teacher_cluster_targets: Sequence[torch.Tensor],
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-layer KL(p_cluster^(T) || softmax(z_router^(S))) over active tokens."""
    if not student_router_logits or not teacher_cluster_targets:
        dev = student_router_logits[0].device if student_router_logits else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=dev)

    mask_f = token_mask.reshape(-1).float() if token_mask is not None else None
    denom = mask_f.sum().clamp_min(1.0) if mask_f is not None else None

    total = torch.zeros((), dtype=torch.float32, device=student_router_logits[0].device)
    count = 0
    for s_r, p_t in zip(student_router_logits, teacher_cluster_targets):
        log_q = F.log_softmax(s_r.float(), dim=-1)
        p = p_t.to(device=s_r.device, dtype=torch.float32)
        log_p = p.clamp_min(1e-12).log()
        kl_tok = (p * (log_p - log_q)).sum(dim=-1)
        if mask_f is not None:
            total = total + (kl_tok * mask_f.to(s_r.device)).sum() / denom.to(s_r.device)
        else:
            total = total + kl_tok.mean()
        count += 1
    return total / max(count, 1)


def select_trainable_params(
    model: nn.Module,
    cfg: RecoveryConfig,
) -> List[Dict]:
    """Configure `requires_grad` and build optimizer param groups."""
    for p in model.parameters():
        p.requires_grad = False

    router_params: List[nn.Parameter] = []
    attn_params: List[nn.Parameter] = []
    main_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if ".base_experts." in name:
            continue
        if cfg.trainable_scope == "moe":
            is_target = (
                ".mlp.gate." in name
                or ".mlp.experts." in name
                or ".post_attention_layernorm." in name
            )
            if not is_target:
                continue
        elif cfg.trainable_scope == "all_no_embed":
            if "embed_tokens" in name or "lm_head" in name:
                continue
        elif cfg.trainable_scope != "all":
            raise ValueError(f"Unknown trainable_scope={cfg.trainable_scope!r}")

        p.requires_grad = True
        if ".mlp.gate." in name:
            router_params.append(p)
        elif ".self_attn." in name or ".input_layernorm." in name or "embed_tokens" in name or "lm_head" in name:
            attn_params.append(p)
        else:
            main_params.append(p)

    groups: List[Dict] = []
    if main_params:
        groups.append({"params": main_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay})
    if attn_params:
        groups.append({"params": attn_params, "lr": cfg.lr * cfg.attn_lr_scale, "weight_decay": cfg.weight_decay})
    if router_params:
        groups.append({
            "params": router_params,
            "lr": cfg.lr * cfg.router_lr_scale,
            "weight_decay": 0.0,
        })
    return groups


class CpuOffloadedAdamW:
    """Minimal zero-dependency fp32 CPU-offloaded AdamW for GPU bf16 parameters.

    Keeps master fp32 weights + m + v on CPU RAM, avoiding DeepSpeed/ROCm JIT
    issues while allowing 3.7B+ models to train comfortably on a single 24 GB GPU.
    """

    def __init__(self, param_groups: List[Dict]):
        self.gpu_groups: List[List[nn.Parameter]] = []
        cpu_groups: List[Dict] = []
        for g in param_groups:
            gpu_ps = list(g["params"])
            self.gpu_groups.append(gpu_ps)
            cpu_ps = [
                nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32).clone())
                for p in gpu_ps
            ]
            cg = {k: v for k, v in g.items() if k != "params"}
            cg["params"] = cpu_ps
            cpu_groups.append(cg)
        self.cpu_opt = torch.optim.AdamW(cpu_groups)

    @property
    def param_groups(self) -> List[Dict]:
        return self.cpu_opt.param_groups

    def zero_grad(self) -> None:
        for gpu_ps in self.gpu_groups:
            for p in gpu_ps:
                p.grad = None
        self.cpu_opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, max_grad_norm: float = 1.0) -> float:
        # Compute global grad norm on GPU first, then copy scaled grads to CPU
        sq_norm = 0.0
        for gpu_ps in self.gpu_groups:
            for p in gpu_ps:
                if p.grad is not None:
                    sq_norm += float(p.grad.detach().float().pow(2).sum().item())
        grad_norm = math.sqrt(sq_norm)
        scale = min(1.0, max_grad_norm / (grad_norm + 1e-6)) if max_grad_norm > 0 else 1.0

        for gpu_ps, cg in zip(self.gpu_groups, self.cpu_opt.param_groups):
            for gp, cp in zip(gpu_ps, cg["params"]):
                if gp.grad is None:
                    cp.grad = None
                else:
                    g_cpu = gp.grad.detach().to(device="cpu", dtype=torch.float32)
                    if scale != 1.0:
                        g_cpu.mul_(scale)
                    cp.grad = g_cpu

        self.cpu_opt.step()

        for gpu_ps, cg in zip(self.gpu_groups, self.cpu_opt.param_groups):
            for gp, cp in zip(gpu_ps, cg["params"]):
                gp.data.copy_(cp.data.to(device=gp.device, dtype=gp.dtype))

        return grad_norm
