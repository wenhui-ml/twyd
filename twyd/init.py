"""LRC warm start: ridge fit of Z_col (Eq.4) + zero-init Z_perp.

Z_col = W_prior + (C^TC + lam I)^{-1} C^T (H^T - C W_prior), lam = tr(C^TC)/r.
C: student compressed layer inputs over calibration set; H^T: matched teacher
activations; W_prior: teacher-weight prior. Layers fit greedily in depth order
(Alg.1); here we expose the single-layer ridge solve + driver stub wired to
ms-swift calibration batches.
"""
from typing import Optional

import torch


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
