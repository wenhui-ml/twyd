"""LRC closed-form & convex solvers for MoE router and expert weight absorption.

1. `fit_cluster_router`:
   Fits student router W_r^(S) in R^(K' x H) to match cluster-aggregated
   teacher routing probabilities P*_j(x) = sum_{e in C_j} P^(T)_e(x) via
   regularized convex KL minimization (with held-out validation safety check).

2. `solve_lrc_down_proj`:
   Solves Coupled Block-Coordinate Gauss-Seidel LRC Ridge Regression over the
   K' surviving experts' `down_proj` matrices so that the compressed MoE layer
   output matches the exact teacher MoE output Y^(T) under the student's actual
   routing weights (automatically curing norm_topk_prob=False scale shifts and
   absorbing pruned experts' functional residuals). Auto-selects Tikhonov
   damping alpha on a held-out calibration validation split, with optional
   low-rank SVD truncation (`lrc_rank`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import MoEAdapter


@dataclass
class RouterFitResult:
    router_weight: torch.Tensor    # [K', H]
    kl_before: float
    kl_after: float
    val_kl_before: float
    val_kl_after: float
    accepted: bool


@dataclass
class SolverResult:
    down_proj: torch.Tensor        # [K', H, d_ff]
    init_fit_rel_mse: float
    init_val_rel_mse: float
    final_fit_rel_mse: float
    final_val_rel_mse: float
    best_alpha: float
    accepted: bool


def aggregate_cluster_probs(
    teacher_probs: torch.Tensor,
    cluster_map: Sequence[int],
    num_survivors: int,
) -> torch.Tensor:
    """Aggregate teacher probs `[N, E]` into cluster probs `[N, K']`."""
    N = teacher_probs.shape[0]
    cmap = torch.tensor(cluster_map, dtype=torch.long, device=teacher_probs.device)
    out = torch.zeros((N, num_survivors), dtype=torch.float32, device=teacher_probs.device)
    out.index_add_(1, cmap, teacher_probs.to(torch.float32))
    out = out.clamp_min(1e-12)
    out = out / out.sum(dim=-1, keepdim=True)
    return out


def _kl_from_logits(target_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean token KL(target_probs || softmax(logits)) in fp32."""
    log_q = F.log_softmax(logits.to(torch.float32), dim=-1)
    log_p = target_probs.clamp_min(1e-12).log()
    return (target_probs * (log_p - log_q)).sum(dim=-1).mean()


def fit_cluster_router(
    teacher_router_weight: torch.Tensor,
    survivors: Sequence[int],
    cluster_map: Sequence[int],
    x_flat: torch.Tensor,
    teacher_probs: torch.Tensor,
    mode: str = "kl_fit",
    val_frac: float = 0.15,
    l2_reg: float = 1e-4,
    lr: float = 5e-2,
    steps: int = 120,
    device: torch.device | str | None = None,
) -> RouterFitResult:
    """Build student router `[K', H]` via slicing or convex mass-preserving KL-clone."""
    K = len(survivors)
    dev = torch.device(device) if device is not None else teacher_router_weight.device
    orig_dtype = teacher_router_weight.dtype

    surv_idx = torch.tensor(survivors, dtype=torch.long, device=teacher_router_weight.device)
    w_slice = teacher_router_weight.index_select(0, surv_idx).detach().to(device=dev, dtype=torch.float32)

    N = x_flat.shape[0]
    n_val = max(1, int(round(N * val_frac))) if N >= 8 else 0
    n_fit = N - n_val

    x_d = x_flat.to(device=dev, dtype=torch.float32)
    p_target = aggregate_cluster_probs(
        teacher_probs.to(device=dev), cluster_map, K
    )

    x_fit, p_fit = x_d[:n_fit], p_target[:n_fit]
    x_val, p_val = (x_d[n_fit:], p_target[n_fit:]) if n_val > 0 else (x_fit, p_fit)

    with torch.no_grad():
        kl_fit_before = float(_kl_from_logits(p_fit, F.linear(x_fit, w_slice)).item())
        kl_val_before = float(_kl_from_logits(p_val, F.linear(x_val, w_slice)).item())

    if mode == "slice" or K == teacher_router_weight.shape[0]:
        return RouterFitResult(
            router_weight=w_slice.to(dtype=orig_dtype),
            kl_before=kl_fit_before,
            kl_after=kl_fit_before,
            val_kl_before=kl_val_before,
            val_kl_after=kl_val_before,
            accepted=False,
        )
    if mode != "kl_fit":
        raise ValueError(f"Unknown router mode={mode!r}; expected 'slice' or 'kl_fit'")

    w_param = nn.Parameter(w_slice.clone())
    opt = torch.optim.Adam([w_param], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.05)

    best_val_kl = kl_val_before
    best_w = w_slice.clone()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits_fit = F.linear(x_fit, w_param)
        kl = _kl_from_logits(p_fit, logits_fit)
        reg = 0.5 * l2_reg * ((w_param - w_slice) ** 2).mean()
        loss = kl + reg
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                vkl = float(_kl_from_logits(p_val, F.linear(x_val, w_param)).item())
                if vkl < best_val_kl:
                    best_val_kl = vkl
                    best_w = w_param.detach().clone()

    with torch.no_grad():
        kl_fit_after = float(_kl_from_logits(p_fit, F.linear(x_fit, best_w)).item())
        kl_val_after = float(_kl_from_logits(p_val, F.linear(x_val, best_w)).item())

    accepted = kl_val_after < kl_val_before
    final_w = best_w if accepted else w_slice
    return RouterFitResult(
        router_weight=final_w.to(dtype=orig_dtype),
        kl_before=kl_fit_before,
        kl_after=kl_fit_after if accepted else kl_fit_before,
        val_kl_before=kl_val_before,
        val_kl_after=kl_val_after if accepted else kl_val_before,
        accepted=accepted,
    )


@torch.no_grad()
def _eval_moe_outputs(
    adapter: MoEAdapter,
    x_flat: torch.Tensor,
    top_w: torch.Tensor,
    top_idx: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: nn.Module,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute compressed MoE output `[N, H]` in fp32 on device."""
    N, H = x_flat.shape
    K = gate_up_proj.shape[0]
    dev = gate_up_proj.device
    w_dtype = gate_up_proj.dtype
    out = torch.zeros((N, H), dtype=torch.float32, device=dev)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xb = x_flat[start:end].to(device=dev, dtype=w_dtype)
        tw = top_w[start:end]
        ti = top_idx[start:end]
        yb = torch.zeros((end - start, H), dtype=torch.float32, device=dev)

        for j in range(K):
            tok_pos, slot_pos = torch.where(ti == j)
            if tok_pos.numel() == 0:
                continue
            h = adapter.expert_hidden(xb[tok_pos], gate_up_proj[j], act_fn)
            y_j = F.linear(h, down_proj[j]).to(torch.float32)
            yb.index_add_(0, tok_pos, y_j * tw[tok_pos, slot_pos].unsqueeze(-1).to(torch.float32))
        out[start:end] = yb
    return out


@torch.no_grad()
def _low_rank_truncate(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Project `delta [H, d_ff]` onto its top-`rank` SVD subspace."""
    if rank <= 0 or rank >= min(delta.shape):
        return delta
    u, s, vh = torch.linalg.svd(delta.to(torch.float32), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


@torch.no_grad()
def _run_coupled_ridge_sweeps(
    expert_h_fit: List[torch.Tensor],
    expert_tok_fit: List[torch.Tensor],
    expert_gram_fit: List[torch.Tensor],
    expert_tr_scale: List[float],
    y_fit_target: torch.Tensor,
    y_fit_init: torch.Tensor,
    down_init: torch.Tensor,
    alpha: float,
    n_sweeps: int,
    damping_step: float,
    lrc_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run coupled Gauss-Seidel LRC Ridge sweeps for a fixed damping `alpha`."""
    K, H, d_ff = down_init.shape
    dev = down_init.device
    down_cur = down_init.to(torch.float32).clone()
    y_cur = y_fit_init.clone()
    eye = torch.eye(d_ff, dtype=torch.float32, device=dev)

    for _ in range(n_sweeps):
        for j in range(K):
            tok_j = expert_tok_fit[j]
            if tok_j.numel() == 0:
                continue
            h_j = expert_h_fit[j]  # [n_j, d_ff] fp32 (already multiplied by router weight)
            r_j = y_fit_target.index_select(0, tok_j) - y_cur.index_select(0, tok_j)  # [n_j, H]
            rhs = h_j.T @ r_j  # [d_ff, H]
            lam = max(alpha * expert_tr_scale[j], 1e-6)
            A = expert_gram_fit[j] + lam * eye
            # Solve A @ dW^T = rhs -> dW = (A^{-1} rhs)^T in [H, d_ff]
            dw_t = torch.linalg.solve(A, rhs)
            dw = dw_t.T
            if lrc_rank > 0:
                dw = _low_rank_truncate(dw, lrc_rank)
            if damping_step != 1.0:
                dw = dw * damping_step
            down_cur[j] = down_cur[j] + dw
            y_cur.index_add_(0, tok_j, h_j @ dw.T)

    return down_cur, y_cur


@torch.no_grad()
def solve_lrc_down_proj(
    adapter: MoEAdapter,
    x_flat: torch.Tensor,
    y_target: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj_init: torch.Tensor,
    act_fn: nn.Module,
    top_k: int,
    norm_topk_prob: bool,
    val_frac: float = 0.15,
    alphas: Sequence[float] = (0.01, 0.1, 0.5, 2.0, 10.0),
    n_sweeps: int = 3,
    damping_step: float = 0.85,
    lrc_rank: int = 0,
    device: torch.device | str | None = None,
) -> SolverResult:
    """Coupled Gauss-Seidel LRC Ridge absorption into surviving experts' `down_proj`."""
    dev = torch.device(device) if device is not None else down_proj_init.device
    orig_dtype = down_proj_init.dtype
    K, H, d_ff = down_proj_init.shape
    N = x_flat.shape[0]

    gate_up_d = gate_up_proj.to(device=dev, dtype=orig_dtype)
    down_d = down_proj_init.to(device=dev, dtype=orig_dtype)
    router_d = router_weight.to(device=dev, dtype=orig_dtype)

    n_val = max(1, int(round(N * val_frac))) if N >= 8 else 0
    n_fit = N - n_val

    x_d = x_flat.to(device=dev, dtype=orig_dtype)
    y_d = y_target.to(device=dev, dtype=torch.float32)

    _, _, top_w, top_idx = adapter.route_with_weight(x_d, router_d, top_k, norm_topk_prob)

    x_fit, y_fit = x_d[:n_fit], y_d[:n_fit]
    tw_fit, ti_fit = top_w[:n_fit], top_idx[:n_fit]
    x_val, y_val = (x_d[n_fit:], y_d[n_fit:]) if n_val > 0 else (x_fit, y_fit)
    tw_val, ti_val = (top_w[n_fit:], top_idx[n_fit:]) if n_val > 0 else (tw_fit, ti_fit)

    y_fit_init = _eval_moe_outputs(adapter, x_fit, tw_fit, ti_fit, gate_up_d, down_d, act_fn)
    y_val_init = (
        _eval_moe_outputs(adapter, x_val, tw_val, ti_val, gate_up_d, down_d, act_fn)
        if n_val > 0
        else y_fit_init
    )

    fit_denom = max(float((y_fit * y_fit).sum().item()), 1e-12)
    val_denom = max(float((y_val * y_val).sum().item()), 1e-12)
    init_fit_rel = float(((y_fit - y_fit_init) ** 2).sum().item()) / fit_denom
    init_val_rel = float(((y_val - y_val_init) ** 2).sum().item()) / val_denom

    # Precompute weighted intermediate activations H_j and Gram A_j = H_j^T H_j once
    expert_h_fit: List[torch.Tensor] = []
    expert_tok_fit: List[torch.Tensor] = []
    expert_gram_fit: List[torch.Tensor] = []
    expert_tr_scale: List[float] = []

    expert_h_val: List[torch.Tensor] = []
    expert_tok_val: List[torch.Tensor] = []

    for j in range(K):
        tok_f, slot_f = torch.where(ti_fit == j)
        expert_tok_fit.append(tok_f)
        if tok_f.numel() > 0:
            h_f = adapter.expert_hidden(x_fit[tok_f], gate_up_d[j], act_fn).to(torch.float32)
            h_f = h_f * tw_fit[tok_f, slot_f].unsqueeze(-1).to(torch.float32)
            A_j = h_f.T @ h_f
            tr_scale = float(torch.diagonal(A_j).mean().item())
            expert_h_fit.append(h_f)
            expert_gram_fit.append(A_j)
            expert_tr_scale.append(max(tr_scale, 1e-6))
        else:
            expert_h_fit.append(torch.empty((0, d_ff), dtype=torch.float32, device=dev))
            expert_gram_fit.append(torch.zeros((d_ff, d_ff), dtype=torch.float32, device=dev))
            expert_tr_scale.append(1.0)

        tok_v, slot_v = torch.where(ti_val == j)
        expert_tok_val.append(tok_v)
        if tok_v.numel() > 0:
            h_v = adapter.expert_hidden(x_val[tok_v], gate_up_d[j], act_fn).to(torch.float32)
            h_v = h_v * tw_val[tok_v, slot_v].unsqueeze(-1).to(torch.float32)
            expert_h_val.append(h_v)
        else:
            expert_h_val.append(torch.empty((0, d_ff), dtype=torch.float32, device=dev))

    best_val_rel = init_val_rel
    best_fit_rel = init_fit_rel
    best_alpha = float("inf")
    best_down = down_d.to(torch.float32)
    accepted = False

    for alpha in alphas:
        cand_down, cand_y_fit = _run_coupled_ridge_sweeps(
            expert_h_fit=expert_h_fit,
            expert_tok_fit=expert_tok_fit,
            expert_gram_fit=expert_gram_fit,
            expert_tr_scale=expert_tr_scale,
            y_fit_target=y_fit,
            y_fit_init=y_fit_init,
            down_init=down_d,
            alpha=float(alpha),
            n_sweeps=n_sweeps,
            damping_step=damping_step,
            lrc_rank=lrc_rank,
        )
        # Fast val evaluation reusing cached expert_h_val
        cand_y_val = torch.zeros_like(y_val)
        for j in range(K):
            tok_v = expert_tok_val[j]
            if tok_v.numel() > 0:
                cand_y_val.index_add_(0, tok_v, expert_h_val[j] @ cand_down[j].T)

        fit_rel = float(((y_fit - cand_y_fit) ** 2).sum().item()) / fit_denom
        val_rel = float(((y_val - cand_y_val) ** 2).sum().item()) / val_denom
        if val_rel < best_val_rel:
            best_val_rel = val_rel
            best_fit_rel = fit_rel
            best_alpha = float(alpha)
            best_down = cand_down
            accepted = True

    return SolverResult(
        down_proj=best_down.to(dtype=orig_dtype),
        init_fit_rel_mse=init_fit_rel,
        init_val_rel_mse=init_val_rel,
        final_fit_rel_mse=best_fit_rel,
        final_val_rel_mse=best_val_rel,
        best_alpha=best_alpha,
        accepted=accepted,
    )
