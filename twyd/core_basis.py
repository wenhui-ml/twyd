"""SVD basis for CORE-LRC.

Gate/up weight W: d_ff x H (d_ff > H).  Full SVD W = U S Vh gives
U: d_ff x d_ff.  T = U[:, :H] spans col(W); U_perp = U[:, H:] spans
col(W)^perp (the stranded complement, Eq.1 fraction f_perp = 1-1/rho).

Down weight W: H x d_ff.  Full SVD W = U S Vh gives Vh: d_ff x d_ff.
Row-space basis = Vh[:H, :]; complement V_perp = Vh[H:, :] and the
reverse construction dW = Z_perp @ V_perp (Table 15 note).

The full SVD runs on CPU fp32 and costs ~2-3 s per projection for a 3B
teacher (8192x2048 / 8192x3072), i.e. 2-5 minutes of GPU-idle startup for a
full model (L*3 projections).  Set ``TWYD_BASIS_CACHE_DIR`` to memoize the
bases on disk; the key is a hash of the teacher weight bytes, so a given
teacher/projection is only decomposed once.
"""
import hashlib
import os
from typing import Literal, Tuple

import torch


def _cache_dir() -> str:
    return os.environ.get("TWYD_BASIS_CACHE_DIR", "") or ""


def _weight_hash(weight: torch.Tensor) -> str:
    t = weight.detach().to("cpu").contiguous()
    try:
        raw = t.view(torch.uint8).numpy().tobytes()
    except Exception:
        raw = t.to(torch.float32).numpy().tobytes()
    h = hashlib.sha256(raw).hexdigest()[:24]
    return f"{h}-{tuple(t.shape)}-{t.dtype}"


def svd_full_cpu(weight: torch.Tensor):
    w = weight.detach().to(torch.float32).cpu()
    return torch.linalg.svd(w, full_matrices=True)


def _cached_basis(weight: torch.Tensor, side: str):
    """Load/save [T U_perp] keyed by the weight content."""
    d = _cache_dir()
    if not d:
        return None
    try:
        from safetensors.torch import load_file, save_file
    except Exception:
        return None
    path = os.path.join(d, f"{side}-{_weight_hash(weight)}.safetensors")
    if os.path.isfile(path):
        try:
            obj = load_file(path)
            return obj["T"].contiguous(), obj["U"].contiguous()
        except Exception:
            return None
    return path  # sentinel: caller computes then saves here


def _save_basis(path: str, T: torch.Tensor, U: torch.Tensor) -> None:
    try:
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file({"T": T.contiguous(), "U": U.contiguous()}, path)
    except Exception:
        pass


def build_gate_basis(weight_dff_x_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d_ff, H = weight_dff_x_H.shape
    assert d_ff > H, f"need wide MLP, got {tuple(weight_dff_x_H.shape)}"
    hit = _cached_basis(weight_dff_x_H, "left")
    if isinstance(hit, tuple):
        return hit
    U, _, _ = svd_full_cpu(weight_dff_x_H)
    T, Up = U[:, :H].contiguous(), U[:, H:].contiguous()
    if isinstance(hit, str):
        _save_basis(hit, T, Up)
    return T, Up


def build_down_basis(weight_H_x_dff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    H, d_ff = weight_H_x_dff.shape
    assert d_ff > H, f"need wide MLP, got {tuple(weight_H_x_dff.shape)}"
    hit = _cached_basis(weight_H_x_dff, "right")
    if isinstance(hit, tuple):
        return hit
    _, _, Vh = svd_full_cpu(weight_H_x_dff)
    R, Vp = Vh[:H, :].contiguous(), Vh[H:, :].contiguous()
    if isinstance(hit, str):
        _save_basis(hit, R, Vp)
    return R, Vp


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
