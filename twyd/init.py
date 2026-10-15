"""LRC warm start: ridge fit of Z_col (Eq.4) + zero-init Z_perp.

Z_col = W_prior + (C^TC + lam I)^{-1} C^T (H^T - C W_prior), lam = tr(C^TC)/r.
C: student compressed layer inputs over calibration set; H^T: matched teacher
activations; W_prior: teacher-weight prior. Layers fit greedily in depth order
(Alg.1); outlier-aware master projection P (Eq.5) for Qwen in build_outlier_aware_P.
"""
from typing import List, Optional

import torch


def build_outlier_aware_P(
    hidden: torch.Tensor, r: int, outlier_keep: int = 0
) -> tuple[torch.Tensor, List[int]]:
    """Master H->r projection P (Eq.5, Fig.10).

    hidden: N x H calibration residual activations (fp32 cpu ok).
    Returns P: r x H, outlier_idx: kept channels (identity rows).
    Llama: outlier_keep=0 -> plain top-r SVD. Qwen: keep k massive-act channels.
    """
    H = hidden.shape[1]
    k = max(0, min(int(outlier_keep), r - 1, H - 1)) if outlier_keep else 0
    out_idx: List[int] = []
    if k > 0:
        peak = hidden.abs().amax(dim=0)
        out_idx = sorted(torch.topk(peak, k).indices.tolist())
    keep = [i for i in range(H) if i not in set(out_idx)]
    rest = hidden[:, keep]
    mu = rest.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd((rest - mu).float().cpu(), full_matrices=False)
    need = r - k
    Vr = Vh[:need, :]  # (r-k) x (H-k)
    P = torch.zeros(r, H)
    for i, ch in enumerate(out_idx):
        P[i, ch] = 1.0
    P[k:, keep] = Vr
    return P.contiguous(), out_idx


def ridge_zcol(C: torch.Tensor, HT: torch.Tensor, W_prior: torch.Tensor,
               ridge_lambda: Optional[float] = None) -> torch.Tensor:
    """All row-major: C: N x r, HT: N x H, W_prior: r x H -> Z_col: r x H."""
    C = C.to(torch.float32)
    HT = HT.to(torch.float32)
    Wp = W_prior.to(torch.float32)
    G = C.T @ C
    r = C.shape[1]
    lam = float(ridge_lambda) if ridge_lambda is not None else float(torch.trace(G) / max(r, 1))
    A = G + lam * torch.eye(r)
    B = C.T @ (HT - C @ Wp)
    D = torch.linalg.solve(A, B)
    return (Wp + D).contiguous()


def zero_perp_(model) -> int:
    n = 0
    for m in model.modules():
        zp = getattr(m, "z_perp", None)
        if isinstance(zp, torch.nn.Parameter):
            with torch.no_grad():
                zp.zero_()
            n += 1
    return n
