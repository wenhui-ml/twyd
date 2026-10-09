"""SVD basis for CORE-LRC.

Gate/up weight W: d_ff x H (d_ff > H).  Full SVD W = U S Vh gives
U: d_ff x d_ff.  T = U[:, :H] spans col(W); U_perp = U[:, H:] spans
col(W)^perp (the stranded complement, Eq.1 fraction f_perp = 1-1/rho).

Down weight W: H x d_ff.  Full SVD W = U S Vh gives Vh: d_ff x d_ff.
Row-space basis = Vh[:H, :]; complement V_perp = Vh[H:, :] and the
reverse construction dW = Z_perp @ V_perp (Table 15 note).
"""
from typing import Literal, Tuple

import torch


def svd_full_cpu(weight: torch.Tensor):
    w = weight.detach().to(torch.float32).cpu()
    return torch.linalg.svd(w, full_matrices=True)


def build_gate_basis(weight_dff_x_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d_ff, H = weight_dff_x_H.shape
    assert d_ff > H, f"need wide MLP, got {tuple(weight_dff_x_H.shape)}"
    U, _, _ = svd_full_cpu(weight_dff_x_H)
    return U[:, :H].contiguous(), U[:, H:].contiguous()


def build_down_basis(weight_H_x_dff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    H, d_ff = weight_H_x_dff.shape
    assert d_ff > H, f"need wide MLP, got {tuple(weight_H_x_dff.shape)}"
    _, _, Vh = svd_full_cpu(weight_H_x_dff)
    return Vh[:H, :].contiguous(), Vh[H:, :].contiguous()


def build_core_basis(weight: torch.Tensor, side: Literal["left", "right"] = "left"):
    if side == "left":
        return build_gate_basis(weight)
    return build_down_basis(weight)


def check_basis(T: torch.Tensor, U_perp: torch.Tensor, tol: float = 1e-4) -> dict:
    """Orthonormality + cross-orthogonality QC for [T U_perp]."""
    TT = T.to(torch.float32)
    UU = U_perp.to(torch.float32)
    eye_T = torch.eye(TT.shape[1])
    eye_U = torch.eye(UU.shape[1])
    return {
        "col_orth_err": (TT.T @ TT - eye_T).abs().max().item(),
        "perp_orth_err": (UU.T @ UU - eye_U).abs().max().item(),
        "cross_err": (TT.T @ UU).abs().max().item(),
        "ok": ((TT.T @ TT - eye_T).abs().max().item() < tol
               and (UU.T @ UU - eye_U).abs().max().item() < tol
               and (TT.T @ UU).abs().max().item() < tol),
    }
