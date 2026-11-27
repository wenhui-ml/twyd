"""Redundancy-aware Top-K' expert selection and functional clustering.

Selection methods:
  - 'freq'        : naive baseline keeping top-K' by router gate_mass
  - 'energy'      : keeps top-K' by weighted output energy G[e, e]
  - 'gram_greedy' : exact OBS/Schur-complement backward elimination on the
                    E x E functional Gram matrix + 1-opt swap polishing
  - 'lrc_cluster' : gram_greedy selection with hybrid functional+Gram
                    clustering of pruned experts to their best host survivor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .profiler import LayerProfile


@dataclass
class SelectionResult:
    method: str
    survivors: List[int]           # sorted or importance-ordered global indices (len K')
    pruned: List[int]              # complement indices (len E - K')
    cluster_map: List[int]         # length E: maps global expert e -> local survivor index in [0..K'-1]
    gram_rel_error: float          # optimal linear Gram reconstruction rel-MSE of subset S
    retained_energy_ratio: float   # sum_{e in S} G[e,e] / sum_e G[e,e]
    retained_mass_ratio: float     # sum_{e in S} mass[e] / sum_e mass[e]


def _gram_subset_residual(
    G: torch.Tensor,
    b: torch.Tensor,
    total_energy: float,
    subset: List[int],
    eps: float = 1e-8,
) -> float:
    """Exact min_alpha ||sum_e Z_e - sum_{j in S} alpha_j Z_j||_F^2."""
    if not subset:
        return total_energy
    idx = torch.tensor(subset, dtype=torch.long)
    G_ss = G.index_select(0, idx).index_select(1, idx)
    b_s = b.index_select(0, idx)
    diag_scale = float(torch.diagonal(G_ss).mean().item()) if len(subset) > 0 else 1.0
    reg = max(eps * max(diag_scale, 1.0), 1e-12)
    G_reg = G_ss + reg * torch.eye(len(subset), dtype=torch.float64)
    alpha = torch.linalg.solve(G_reg, b_s)
    explained = float((b_s @ alpha).item())
    return max(total_energy - explained, 0.0)


def _obs_backward_select(
    G: torch.Tensor,
    target_k: int,
    eps: float = 1e-8,
    max_swap_rounds: int = 10,
) -> List[int]:
    """Exact OBS/Schur backward elimination on E x E Gram matrix + 1-opt swap polish."""
    E = G.shape[0]
    b = G.sum(dim=1)  # G @ 1
    total_energy = float(b.sum().item())
    diag_scale = float(torch.diagonal(G).mean().item())
    reg = max(eps * max(diag_scale, 1.0), 1e-12)

    active = list(range(E))
    while len(active) > target_k:
        idx = torch.tensor(active, dtype=torch.long)
        G_aa = G.index_select(0, idx).index_select(1, idx)
        b_a = b.index_select(0, idx)
        G_reg = G_aa + reg * torch.eye(len(active), dtype=torch.float64)
        M = torch.linalg.inv(G_reg)
        alpha = M @ b_a
        # Exact increase in residual upon setting alpha[i] = 0: alpha[i]^2 / M[i,i]
        diag_m = torch.diagonal(M).clamp_min(1e-18)
        delta_err = (alpha * alpha) / diag_m
        drop_pos = int(torch.argmin(delta_err).item())
        active.pop(drop_pos)

    # 1-opt swap polishing to escape greedy local optima
    cur_err = _gram_subset_residual(G, b, total_energy, active, eps=eps)
    for _ in range(max_swap_rounds):
        improved = False
        active_set = set(active)
        inactive = [e for e in range(E) if e not in active_set]
        if not inactive:
            break

        # Find weakest current member via OBS delta
        idx = torch.tensor(active, dtype=torch.long)
        G_aa = G.index_select(0, idx).index_select(1, idx)
        b_a = b.index_select(0, idx)
        G_reg = G_aa + reg * torch.eye(len(active), dtype=torch.float64)
        M = torch.linalg.inv(G_reg)
        alpha = M @ b_a
        order_weak = torch.argsort((alpha * alpha) / torch.diagonal(M).clamp_min(1e-18))

        best_swap = None
        best_err = cur_err * (1.0 - 1e-9)
        # Check the 8 weakest active members against all inactive candidates
        for pos_t in order_weak[: min(8, len(active))]:
            pos = int(pos_t.item())
            base = active[:pos] + active[pos + 1 :]
            for cand in inactive:
                trial = base + [cand]
                err = _gram_subset_residual(G, b, total_energy, trial, eps=eps)
                if err < best_err:
                    best_err = err
                    best_swap = (pos, cand)

        if best_swap is not None:
            pos, cand = best_swap
            active[pos] = cand
            cur_err = best_err
            improved = True
        if not improved:
            break

    active.sort()
    return active


def _build_cluster_map(
    profile: LayerProfile,
    survivors: List[int],
    gram_weight: float = 0.25,
) -> List[int]:
    """Assign each expert e in [0..E-1] to its closest surviving host in [0..K'-1]."""
    E = profile.num_experts
    G = profile.gram.to(torch.float64)
    diag_sqrt = torch.diagonal(G).clamp_min(1e-18).sqrt()
    gram_cos = (G / (diag_sqrt.unsqueeze(1) * diag_sqrt.unsqueeze(0))).to(torch.float32)
    sim = profile.func_sim.to(torch.float32) + float(gram_weight) * gram_cos

    surv_to_local = {s: j for j, s in enumerate(survivors)}
    surv_idx = torch.tensor(survivors, dtype=torch.long)

    cluster_map: List[int] = []
    for e in range(E):
        if e in surv_to_local:
            cluster_map.append(surv_to_local[e])
        else:
            scores = sim[e].index_select(0, surv_idx)
            best_local = int(torch.argmax(scores).item())
            cluster_map.append(best_local)
    return cluster_map


def select_top_k_experts(
    profile: LayerProfile,
    target_experts: int,
    method: str = "lrc_cluster",
    gram_weight: float = 0.25,
) -> SelectionResult:
    """Select the `target_experts` most accurate experts and cluster pruned ones."""
    E = profile.num_experts
    K = int(target_experts)
    if not (1 <= K <= E):
        raise ValueError(f"target_experts={K} must be in [1, {E}]")

    G = profile.gram.to(torch.float64)
    b = G.sum(dim=1)
    total_energy = max(float(b.sum().item()), 1e-18)

    if K == E:
        survivors = list(range(E))
    elif method == "freq":
        top = torch.topk(profile.gate_mass, K).indices.tolist()
        survivors = sorted(top)
    elif method == "energy":
        top = torch.topk(profile.output_energy, K).indices.tolist()
        survivors = sorted(top)
    elif method in ("gram_greedy", "lrc_cluster"):
        survivors = _obs_backward_select(G, K)
    else:
        raise ValueError(
            f"Unknown selection method={method!r}; expected 'freq'|'energy'|'gram_greedy'|'lrc_cluster'"
        )

    surv_set = set(survivors)
    pruned = [e for e in range(E) if e not in surv_set]
    cluster_map = _build_cluster_map(
        profile,
        survivors,
        gram_weight=gram_weight if method == "lrc_cluster" else 0.0,
    )

    gram_res = _gram_subset_residual(G, b, total_energy, survivors)
    gram_rel_err = gram_res / total_energy

    diag_sum = float(profile.output_energy.sum().item())
    mass_sum = float(profile.gate_mass.sum().item())
    surv_idx = torch.tensor(survivors, dtype=torch.long)
    ret_energy = float(profile.output_energy.index_select(0, surv_idx).sum().item()) / max(diag_sum, 1e-18)
    ret_mass = float(profile.gate_mass.index_select(0, surv_idx).sum().item()) / max(mass_sum, 1e-18)

    return SelectionResult(
        method=method,
        survivors=survivors,
        pruned=pruned,
        cluster_map=cluster_map,
        gram_rel_error=gram_rel_err,
        retained_energy_ratio=ret_energy,
        retained_mass_ratio=ret_mass,
    )
