"""LRC warm start: ridge fit of Z_col (Eq.4) + optional residual Z_perp.

Two equivalent views are implemented:

* ``ridge_zcol``: paper Eq.4 in the factorized coordinates (Z_col: r x H).
* ``ridge_deployed``: the same least-squares fit on the deployed weight
  (W: out x in), which is what the calibration loop can accumulate streaming.

``calibrate_ridge`` runs the greedy, data-aware initialization of Alg.1 on any
student/teacher pair with matching MLP module names.  It keeps only sufficient
statistics ``C^T C`` / ``C^T H_T`` per projection, so calibration memory is
O(in^2 + in*out) and independent of the number of calibration tokens.

The result is *calibration-distribution* equivalence (least squares on the
calibration activations), not exact function equivalence for every input: the
width-r student cannot represent the teacher's full H-dimensional residual
stream.  That residual is exactly what ``Z_perp`` and later training address.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .core_basis import build_core_basis


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


def ridge_deployed(C: torch.Tensor, HT: torch.Tensor,
                   W_prior: Optional[torch.Tensor] = None,
                   ridge_lambda: Optional[float] = None):
    """Least-squares fit of the *deployed* weight W: out x in.

    C: N x in student inputs, HT: N x out teacher targets.
    With W_prior: W = W_prior + ((C^T C + lam I)^-1 C^T (HT - C W_prior^T))^T.
    Returns (W, stats) where stats has rel_err_before/after, n_tokens.
    """
    C = C.to(torch.float32)
    HT = HT.to(torch.float32)
    n, in_f = C.shape
    out_f = HT.shape[1]
    Wp = (torch.zeros(out_f, in_f) if W_prior is None
          else W_prior.detach().to(torch.float32).cpu())
    assert tuple(Wp.shape) == (out_f, in_f), (tuple(Wp.shape), (out_f, in_f))
    G = C.T @ C
    B = C.T @ HT
    sst = float((HT * HT).sum())
    return _solve_from_stats(G, B, sst, n, Wp, ridge_lambda)


def _solve_from_stats(G: torch.Tensor, B: torch.Tensor, sst: float,
                      n_tokens: int, W_prior: torch.Tensor,
                      ridge_lambda: Optional[float] = None):
    """Solve the ridge normal equations from streaming statistics."""
    in_f, out_f = G.shape[0], B.shape[1]
    lam = float(ridge_lambda) if ridge_lambda is not None else float(torch.trace(G) / max(in_f, 1))
    A = G + lam * torch.eye(in_f)
    rhs = B - G @ W_prior.T
    try:
        delta = torch.linalg.solve(A, rhs)
    except Exception:
        delta = torch.linalg.lstsq(A, rhs).solution
    W = W_prior + delta.T
    resid_before = sst - 2.0 * float((W_prior * B.T).sum()) \
        + float(((W_prior @ G) * W_prior).sum())
    resid_after = sst - 2.0 * float((W * B.T).sum()) + float(((W @ G) * W).sum())
    stats = {
        "n_tokens": int(n_tokens),
        "lambda": lam,
        "rel_err_before": (max(resid_before, 0.0) / max(sst, 1e-30)) ** 0.5,
        "rel_err_after": (max(resid_after, 0.0) / max(sst, 1e-30)) ** 0.5,
    }
    return W.contiguous(), stats


def factorize_deployed(W: torch.Tensor, teacher_weight: torch.Tensor,
                       side: str = "left"):
    """Split deployed W into (Z_col, Z_perp, basis).

    left  (gate/up): W: d_ff x r  -> Z_col: r x H,     Z_perp: r x (d_ff-H)
    right (down):    W: r x d_ff  -> Z_col: r x H,     Z_perp: r x (d_ff-H)

    The split is exact because [T U_perp] (or [R; V_perp]) is orthogonal.
    """
    W = W.detach().to(torch.float32).cpu()
    if side == "left":
        d_ff, r = W.shape
        T, U = build_core_basis(teacher_weight, "left")
        assert T.shape == (d_ff, teacher_weight.shape[1])
        Z_col = (T.T @ W).T.contiguous()
        W_col = T @ Z_col.T
        Z_perp = (U.T @ (W - W_col)).T.contiguous()
        return Z_col, Z_perp, (T, U)
    r, d_ff = W.shape
    R, V = build_core_basis(teacher_weight, "right")
    Z_col = (W @ R.T).contiguous()
    W_col = Z_col @ R
    Z_perp = ((W - W_col) @ V.T).contiguous()
    return Z_col, Z_perp, (R, V)


def build_master_P(teacher: nn.Module, r: int) -> torch.Tensor:
    """Master residual-stream projection P: r x H from the tied embedding SVD.

    Paper Alg.1 step (i): P = Vh[:r] of the (tied) embedding matrix.  Rows are
    orthonormal, so P P^T = I_r.
    """
    emb = teacher.get_input_embeddings().weight.detach().float().cpu()
    if emb.shape[1] < r:
        raise ValueError(f"embedding hidden {emb.shape[1]} < r={r}")
    _, _, Vh = torch.linalg.svd(emb, full_matrices=False)
    return Vh[:r, :].contiguous()


def _layer_prefix(model: nn.Module) -> str:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer.h"
    raise ValueError(f"unsupported model layout for P projection: {type(model).__name__}")


def _module_opt(model: nn.Module, name: str):
    try:
        parent, attr = name.rsplit(".", 1)
        return getattr(model.get_submodule(parent), attr)
    except Exception:
        return None


def head_projection_mats(tcfg, r: int, P: torch.Tensor,
                         head_mode: str = "select"):
    """Build S_q (r x H) and S_kv (new_kv*d x kv*d) for width-r attention.

    select: keep the first new_heads teacher q heads / kv_map teacher kv heads.
    merge:  average groups of teacher heads (requires exact divisibility).
    """
    from .tuner import _student_head_layout
    H = int(P.shape[1])
    heads, kv, head_dim, new_heads, new_kv, kv_map = _student_head_layout(tcfg, r)
    if new_heads * head_dim != r:
        raise ValueError(f"new_heads*head_dim={new_heads * head_dim} != r={r}")
    eye_d = torch.eye(head_dim)
    S_q = torch.zeros(r, H)
    gq = heads // new_heads if new_heads and heads % new_heads == 0 else 0
    if head_mode == "merge" and gq > 1:
        for i in range(new_heads):
            for j in range(gq):
                S_q[i * head_dim:(i + 1) * head_dim,
                    (i * gq + j) * head_dim:(i * gq + j + 1) * head_dim] = eye_d / gq
    else:
        for i in range(new_heads):
            S_q[i * head_dim:(i + 1) * head_dim, i * head_dim:(i + 1) * head_dim] = eye_d
    S_kv = torch.zeros(new_kv * head_dim, kv * head_dim)
    gk = kv // new_kv if new_kv and kv % new_kv == 0 else 0
    if head_mode == "merge" and gk > 1:
        for j in range(new_kv):
            for t in range(gk):
                S_kv[j * head_dim:(j + 1) * head_dim,
                     (j * gk + t) * head_dim:(j * gk + t + 1) * head_dim] = eye_d / gk
    else:
        for j, src in enumerate(kv_map):
            S_kv[j * head_dim:(j + 1) * head_dim, src * head_dim:(src + 1) * head_dim] = eye_d
    return S_q.contiguous(), S_kv.contiguous()


def project_student_with_P_(student: nn.Module, teacher: nn.Module,
                            P: torch.Tensor, head_mode: str = "select") -> int:
    """Project teacher weights onto the master P (head-aware for attention).

    Residual stream: x_S = P x_T.
      embed:   E_S = E_T P^T
      q:       W_qS = S_q W_qT P^T
      k/v:     W_kS = S_kv W_kT P^T
      o:       W_oS = P W_oT S_q^T
      gate/up: W_S = W_T P^T
      down:    W_S = P W_T
      norms:   gamma_S = P gamma_T
    S_q selects the first new_heads teacher q heads; S_kv follows kv_map, so
    GQA grouping is preserved (same mapping as _shrink_student_).
    """
    P = P.detach().float().cpu()
    r, H = P.shape
    tcfg = getattr(teacher.config, "text_config", None) or teacher.config
    S_q, S_kv = head_projection_mats(tcfg, r, P, head_mode)
    pref_t, pref_s = _layer_prefix(teacher), _layer_prefix(student)
    n = 0

    def _proj_weight(s_mod, t_mod, left=None, right=None):
        nonlocal n
        if s_mod is None or t_mod is None or not hasattr(t_mod, "weight"):
            return
        W = t_mod.weight.detach().float().cpu()
        if left is not None:
            W = left @ W
        if right is not None:
            W = W @ right
        with torch.no_grad():
            s_mod.weight.copy_(W.to(device=s_mod.weight.device, dtype=s_mod.weight.dtype))
        if getattr(t_mod, "bias", None) is not None and getattr(s_mod, "bias", None) is not None:
            b = t_mod.bias.detach().float().cpu()
            if left is not None:
                b = left @ b
            with torch.no_grad():
                s_mod.bias.copy_(b.to(device=s_mod.bias.device, dtype=s_mod.bias.dtype))
        n += 1

    e_s, e_t = student.get_input_embeddings(), teacher.get_input_embeddings()
    with torch.no_grad():
        e_s.weight.copy_((e_t.weight.detach().float().cpu() @ P.T).to(
            device=e_s.weight.device, dtype=e_s.weight.dtype))
    n += 1
    tied = bool(getattr(getattr(student.config, "text_config", None) or student.config,
                        "tie_word_embeddings", False))
    out_s, out_t = student.get_output_embeddings(), teacher.get_output_embeddings()
    if not tied and out_s is not None and out_t is not None:
        with torch.no_grad():
            out_s.weight.copy_((out_t.weight.detach().float().cpu() @ P.T).to(
                device=out_s.weight.device, dtype=out_s.weight.dtype))
        n += 1

    for l in range(int(getattr(tcfg, "num_hidden_layers", 0))):
        for attn in ("self_attn", "attention"):
            q_t = _module_opt(teacher, f"{pref_t}.{l}.{attn}.q_proj")
            if q_t is None:
                continue
            q_s = _module_opt(student, f"{pref_s}.{l}.{attn}.q_proj")
            k_s = _module_opt(student, f"{pref_s}.{l}.{attn}.k_proj")
            v_s = _module_opt(student, f"{pref_s}.{l}.{attn}.v_proj")
            o_sa = _module_opt(student, f"{pref_s}.{l}.{attn}.o_proj")
            k_t = _module_opt(teacher, f"{pref_t}.{l}.{attn}.k_proj")
            v_t = _module_opt(teacher, f"{pref_t}.{l}.{attn}.v_proj")
            o_t = _module_opt(teacher, f"{pref_t}.{l}.{attn}.o_proj")
            _proj_weight(q_s, q_t, left=S_q, right=P.T)
            _proj_weight(k_s, k_t, left=S_kv, right=P.T)
            _proj_weight(v_s, v_t, left=S_kv, right=P.T)
            _proj_weight(o_sa, o_t, left=P, right=S_q.T)
            break
        g_t = _module_opt(teacher, f"{pref_t}.{l}.mlp.gate_proj")
        if g_t is not None:
            _proj_weight(_module_opt(student, f"{pref_s}.{l}.mlp.gate_proj"),
                         g_t, right=P.T)
            _proj_weight(_module_opt(student, f"{pref_s}.{l}.mlp.up_proj"),
                         _module_opt(teacher, f"{pref_t}.{l}.mlp.up_proj"),
                         right=P.T)
            d_t = _module_opt(teacher, f"{pref_t}.{l}.mlp.down_proj")
            if d_t is not None:
                _proj_weight(_module_opt(student, f"{pref_s}.{l}.mlp.down_proj"),
                             d_t, left=P)
        for ln in ("input_layernorm", "post_attention_layernorm"):
            s_ln = _module_opt(student, f"{pref_s}.{l}.{ln}")
            t_ln = _module_opt(teacher, f"{pref_t}.{l}.{ln}")
            if s_ln is not None and t_ln is not None and hasattr(t_ln, "weight"):
                with torch.no_grad():
                    s_ln.weight.copy_((P @ t_ln.weight.detach().float().cpu()).to(
                        device=s_ln.weight.device, dtype=s_ln.weight.dtype))
                n += 1
    s_fn = _module_opt(student, "model.norm") or _module_opt(student, "transformer.ln_f")
    t_fn = _module_opt(teacher, "model.norm") or _module_opt(teacher, "transformer.ln_f")
    if s_fn is not None and t_fn is not None and hasattr(t_fn, "weight"):
        with torch.no_grad():
            s_fn.weight.copy_((P @ t_fn.weight.detach().float().cpu()).to(
                device=s_fn.weight.device, dtype=s_fn.weight.dtype))
        n += 1
    try:
        student.tie_weights()
    except Exception:
        pass
    return n


def _is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or callable(getattr(m, "merged_weight", None)) \
        or hasattr(m, "z_col") or hasattr(m, "base_weight")


def _deployed_weight(m: nn.Module) -> torch.Tensor:
    mw = getattr(m, "merged_weight", None)
    if callable(mw):
        return mw().detach()
    bw = getattr(m, "base_weight", None)
    if isinstance(bw, torch.Tensor):
        return bw.detach()
    return m.weight.detach()


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _flat(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    return x.reshape(-1, x.shape[-1])


def collect_mlp_groups(student: nn.Module, teacher: nn.Module,
                       target_modules: Sequence[str] = ("gate_proj", "up_proj", "down_proj")
                       ) -> List[dict]:
    """Pair student/teacher MLP linears by dotted name.

    Works for plain nn.Linear students and for factorized CoreLinear/
    DenseLinearLoRA students (anything exposing merged_weight / z_col).
    """
    tmods = dict(teacher.named_modules())
    groups: Dict[str, dict] = {}
    for name, mod in student.named_modules():
        short = name.rsplit(".", 1)[-1]
        if short not in target_modules and name not in target_modules:
            continue
        if not _is_linear_like(mod):
            continue
        tmod = tmods.get(name)
        if tmod is None or not hasattr(tmod, "weight"):
            continue
        base = name.rsplit(".", 1)[0] if "." in name else ""
        groups.setdefault(base, {})[short] = (mod, tmod)
    from .config import layer_idx_of
    out = []
    for base, projs in groups.items():
        if not projs:
            continue
        out.append({"base": base, "layer": layer_idx_of(base), "projs": projs})
    out.sort(key=lambda g: (g["layer"] if g["layer"] is not None else 10**9, g["base"]))
    return out


def _forward_capture(model: nn.Module, batch: dict,
                     specs: Dict[str, Tuple[nn.Module, str]]) -> Dict[str, torch.Tensor]:
    """Run one forward and capture pre/post activations for named modules."""
    store: Dict[str, torch.Tensor] = {}
    handles = []
    for key, (mod, mode) in specs.items():
        if mode == "pre":
            def _pre(_m, args, _key=key):
                x = _first_tensor(args)
                if x is not None:
                    store[_key] = _flat(x)
            handles.append(mod.register_forward_pre_hook(_pre))
        else:
            def _post(_m, _args, out, _key=key):
                x = _first_tensor(out)
                if x is not None:
                    store[_key] = _flat(x)
            handles.append(mod.register_forward_hook(_post))
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    try:
        with torch.no_grad():
            try:
                model(**b, use_cache=False)
            except TypeError:
                model(**b)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in specs if k not in store or store[k].numel() == 0]
    if missing:
        raise RuntimeError(f"failed to capture activations for {missing}")
    return store


def _set_lora_perp(m: nn.Module, Z_perp: torch.Tensor, rank: int, scaling: float) -> None:
    k = int(min(rank, Z_perp.shape[0], Z_perp.shape[1]))
    if k <= 0:
        with torch.no_grad():
            m.lora_A.zero_()
            m.lora_B.zero_()
        return
    U, S, Vh = torch.linalg.svd(Z_perp.to(torch.float32), full_matrices=False)
    scale = max(float(scaling), 1e-12)
    s = torch.sqrt(S[:k].clamp_min(0.0) / scale)
    A = (U[:, :k] * s).to(device=m.lora_A.device, dtype=m.lora_A.dtype)
    B = (Vh[:k, :].T * s).to(device=m.lora_B.device, dtype=m.lora_B.dtype)
    with torch.no_grad():
        m.lora_A.copy_(A)
        m.lora_B.copy_(B)


def _factorize_with_basis(W: torch.Tensor, basis_col: torch.Tensor,
                          basis_perp: torch.Tensor, side: str = "left"):
    """Split W using an already-computed orthonormal basis (no SVD)."""
    dev = basis_col.device
    W = W.detach().to(device=dev, dtype=torch.float32)
    T = basis_col.detach().to(torch.float32)
    U = basis_perp.detach().to(torch.float32)
    if side == "left":
        Z_col = (T.T @ W).T.contiguous()
        Z_perp = (U.T @ (W - T @ Z_col.T)).T.contiguous()
    else:
        Z_col = (W @ T.T).contiguous()
        Z_perp = ((W - Z_col @ T) @ U.T).contiguous()
    return Z_col, Z_perp


def set_deployed_weight(m: nn.Module, W: torch.Tensor, teacher_weight: torch.Tensor,
                        side: str = "left", perp: str = "residual",
                        lora_rank: int = 0) -> Dict[str, float]:
    """Write a fitted deployed weight back into a plain/factorized linear."""
    W = W.detach().to(torch.float32).cpu()
    out = {}
    if isinstance(m, nn.Linear) and not hasattr(m, "base_weight"):
        with torch.no_grad():
            m.weight.copy_(W.to(device=m.weight.device, dtype=m.weight.dtype))
        return out
    bw = getattr(m, "base_weight", None)
    if isinstance(bw, torch.Tensor) and not hasattr(m, "z_col"):
        with torch.no_grad():
            bw.copy_(W.to(device=bw.device, dtype=bw.dtype))
            if getattr(m, "lora_B", None) is not None:
                m.lora_B.zero_()
        return out
    if not hasattr(m, "z_col"):
        with torch.no_grad():
            m.weight.copy_(W.to(device=m.weight.device, dtype=m.weight.dtype))
        return out
    bc = getattr(m, "basis_col", None)
    bp = getattr(m, "basis_perp", None)
    if not getattr(m, "with_perp", False):
        if isinstance(bc, torch.Tensor) and bc.numel():
            Wb = W.to(device=bc.device, dtype=torch.float32)
            Z_col = ((bc.T.float() @ Wb).T if side == "left" else (Wb @ bc.float().T)).contiguous()
        else:
            Z_col, _, _ = factorize_deployed(W, teacher_weight, side)
        with torch.no_grad():
            m.z_col.copy_(Z_col.to(device=m.z_col.device, dtype=m.z_col.dtype))
        out["z_col_norm"] = float(Z_col.norm())
        return out
    if isinstance(bc, torch.Tensor) and bc.numel() and isinstance(bp, torch.Tensor) and bp.numel():
        Z_col, Z_perp = _factorize_with_basis(W, bc, bp, side)
    else:
        Z_col, Z_perp, _ = factorize_deployed(W, teacher_weight, side)
    with torch.no_grad():
        m.z_col.copy_(Z_col.to(device=m.z_col.device, dtype=m.z_col.dtype))
    out["z_col_norm"] = float(Z_col.norm())
    if perp == "zero":
        Z_perp = torch.zeros_like(Z_perp)
    out["z_perp_norm"] = float(Z_perp.norm())
    lora_a = getattr(m, "lora_A", None)
    if isinstance(lora_a, nn.Parameter):
        if perp == "zero":
            with torch.no_grad():
                m.lora_A.zero_()
                m.lora_B.zero_()
        else:
            _set_lora_perp(m, Z_perp, lora_rank or getattr(m, "lora_rank", 0),
                           getattr(m, "scaling", 1.0))
        return out
    zp = getattr(m, "z_perp", None)
    if isinstance(zp, nn.Parameter):
        with torch.no_grad():
            zp.copy_(Z_perp.to(device=zp.device, dtype=zp.dtype))
    return out


def _as_batches(input_ids, batch_size: int):
    if isinstance(input_ids, dict):
        n = len(next(iter(input_ids.values())))
        for i in range(0, n, batch_size):
            yield {k: (v[i:i + batch_size] if isinstance(v, torch.Tensor) else v)
                   for k, v in input_ids.items()}
    elif isinstance(input_ids, torch.Tensor):
        for i in range(0, input_ids.shape[0], batch_size):
            yield {"input_ids": input_ids[i:i + batch_size]}
    else:
        for x in input_ids:
            yield {"input_ids": x}


def _accumulate(student, teacher, batches, names, projs,
                target_proj: Optional[torch.Tensor] = None):
    """Stream C^T C, C^T H_T and ||H_T||^2 for a set of projections."""
    acc = {n: {"G": None, "B": None, "sst": 0.0, "n": 0} for n in names}
    for batch in batches:
        s_specs = {n: (projs[n][0], "pre") for n in names}
        t_specs = {n: (projs[n][1], "post") for n in names}
        s_store = _forward_capture(student, batch, s_specs)
        t_store = _forward_capture(teacher, batch, t_specs)
        for n in names:
            C, HT = s_store[n], t_store[n]
            m = min(C.shape[0], HT.shape[0])
            C, HT = C[:m], HT[:m]
            # down_proj: teacher output is H, student output is r.  The width-r
            # student coordinate system is the slice convention (first r dims),
            # matching _shrink_student_ and check_init.  A learned master P can
            # be passed later; the compression residual is exactly the target of
            # Z_perp/pretraining.
            out_dim = _deployed_weight(projs[n][0]).shape[0]
            if HT.shape[1] != out_dim:
                if target_proj is not None:
                    P = target_proj.to(device=HT.device, dtype=torch.float32)
                    if tuple(P.shape) != (HT.shape[1], out_dim):
                        raise RuntimeError(
                            f"{n}: target_proj {tuple(P.shape)} != ({HT.shape[1]}, {out_dim})")
                    HT = HT @ P
                elif HT.shape[1] > out_dim:
                    HT = HT[:, :out_dim]
                else:
                    raise RuntimeError(f"{n}: teacher out {HT.shape[1]} < student out {out_dim}")
            if acc[n]["G"] is None:
                acc[n]["G"] = torch.zeros(C.shape[1], C.shape[1])
                acc[n]["B"] = torch.zeros(C.shape[1], HT.shape[1])
            acc[n]["G"] += C.T @ C
            acc[n]["B"] += C.T @ HT
            acc[n]["sst"] += float((HT * HT).sum())
            acc[n]["n"] += int(m)
    return acc


def _accumulate_attn(student, teacher, batches, names, projs, target_projs):
    """Stream ridge stats for q/k/v/o; teacher targets are head-projected."""
    acc = {n: {"G": None, "B": None, "sst": 0.0, "n": 0} for n in names}
    for batch in batches:
        s_store = _forward_capture(student, batch, {n: (projs[n][0], "pre") for n in names})
        t_store = _forward_capture(teacher, batch, {n: (projs[n][1], "post") for n in names})
        for n in names:
            C = s_store[n]
            HT = t_store[n]
            proj = target_projs.get(n)
            if proj is not None:
                P = proj.to(device=HT.device, dtype=torch.float32)
                out_dim = _deployed_weight(projs[n][0]).shape[0]
                if tuple(P.shape) != (out_dim, HT.shape[1]):
                    raise RuntimeError(
                        f"{n}: target_proj {tuple(P.shape)} != ({out_dim}, {HT.shape[1]})")
                HT = HT @ P.T
            m = min(C.shape[0], HT.shape[0])
            C, HT = C[:m], HT[:m]
            if acc[n]["G"] is None:
                acc[n]["G"] = torch.zeros(C.shape[1], C.shape[1])
                acc[n]["B"] = torch.zeros(C.shape[1], HT.shape[1])
            acc[n]["G"] += C.T @ C
            acc[n]["B"] += C.T @ HT
            acc[n]["sst"] += float((HT * HT).sum())
            acc[n]["n"] += int(m)
    return acc


def calibrate_attention(student: nn.Module, teacher: nn.Module, input_ids, *,
                        P: torch.Tensor, head_mode: str = "select",
                        ridge_lambda: Optional[float] = None,
                        batch_size: int = 2, log=print) -> dict:
    """Per-layer ridge for q/k/v/o, with head-projected teacher targets.

    q:  target S_q q_T   (r)
    k/v: target S_kv k_T (new_kv*head_dim)
    o:  target P o_T     (r)
    q/k/v are fit and updated first; o is then fit on the updated attention
    output.  This is the attention analogue of the MLP col-path ridge.
    """
    P = P.detach().float().cpu()
    r, H = P.shape
    tcfg = getattr(teacher.config, "text_config", None) or teacher.config
    S_q, S_kv = head_projection_mats(tcfg, r, P, head_mode)
    target_projs = {"q_proj": S_q, "k_proj": S_kv, "v_proj": S_kv, "o_proj": P}
    groups = collect_mlp_groups(student, teacher, ("q_proj", "k_proj", "v_proj", "o_proj"))
    if not groups:
        raise RuntimeError("no matching attention projections found")
    was_training = student.training
    student.eval()
    teacher.eval()
    rows: List[dict] = []
    try:
        for g in groups:
            projs = g["projs"]
            for pass_names in (["q_proj", "k_proj", "v_proj"], ["o_proj"]):
                names = [n for n in pass_names if n in projs]
                if not names:
                    continue
                acc = _accumulate_attn(student, teacher,
                                       _as_batches(input_ids, batch_size),
                                       names, projs, target_projs)
                for n in names:
                    s_mod, t_mod = projs[n]
                    Wp = _deployed_weight(s_mod).float().cpu()
                    W, st = _solve_from_stats(acc[n]["G"], acc[n]["B"], acc[n]["sst"],
                                              acc[n]["n"], Wp, ridge_lambda)
                    set_deployed_weight(s_mod, W, t_mod.weight, side="left", perp="zero")
                    We = _deployed_weight(s_mod).detach().float().cpu()
                    resid = (acc[n]["sst"] - 2.0 * float((We * acc[n]["B"].T).sum())
                             + float(((We @ acc[n]["G"]) * We).sum()))
                    rel_deployed = (max(resid, 0.0) / max(acc[n]["sst"], 1e-30)) ** 0.5
                    row = {"layer": g["layer"], "base": g["base"], "proj": n,
                           **st, "rel_err_deployed": rel_deployed}
                    rows.append(row)
                    if log is not None:
                        log(f"[attn] L{g['layer']} {n:7s} rel "
                            f"{st['rel_err_before']:.4f} -> {rel_deployed:.4f}")
    finally:
        student.train(was_training)
    valid = [x for x in rows if x["rel_err_before"] > 0]
    return {
        "rows": rows,
        "n_groups": len(groups),
        "n_tokens": max((x["n_tokens"] for x in rows), default=0),
        "head_mode": head_mode,
        "mean_rel_before": sum(x["rel_err_before"] for x in valid) / max(len(valid), 1),
        "mean_rel_deployed": sum(x["rel_err_deployed"] for x in valid) / max(len(valid), 1),
    }


def calibrate_ridge(student: nn.Module, teacher: nn.Module, input_ids, *,
                    target_modules: Sequence[str] = ("gate_proj", "up_proj", "down_proj"),
                    ridge_lambda: Optional[float] = None,
                    perp: str = "residual",
                    joint: bool = True,
                    batch_size: int = 2,
                    target_proj: Optional[torch.Tensor] = None,
                    log=print) -> dict:
    """Greedy data-aware ridge init (paper Alg.1) on a live student.

    Fits the deployed MLP weights layer by layer, on the student's own
    compressed inputs, then writes them back as Z_col (+ optional residual
    Z_perp).  Only sufficient statistics are kept, so memory is independent of
    the number of calibration tokens.

    perp:
      "zero"     -> Z_perp = 0 (paper default; exact plain-LRC at step 0)
      "residual" -> Z_perp = least-squares teacher residual in U_perp
      "lowrank"  -> same, but CoreLinearLoRA keeps the top-k SVD directions
    joint:
      True  -> gate/up/down fitted in one pass (fast; down sees pre-update MLP)
      False -> gate/up fitted, then down re-run on the updated MLP (greedy)
    """
    if perp not in ("zero", "residual", "lowrank"):
        raise ValueError(f"perp={perp}")
    groups = collect_mlp_groups(student, teacher, target_modules)
    if not groups:
        raise RuntimeError("no matching MLP projections found for ridge calibration")
    was_training = student.training
    student.eval()
    teacher.eval()
    rows: List[dict] = []
    try:
        for g in groups:
            projs = g["projs"]
            down_names = [n for n in projs
                          if "down" in n.lower() or n in ("w2", "c_proj")]
            up_names = [n for n in projs if n not in down_names]
            passes = []
            if joint:
                passes.append(up_names + down_names)
            else:
                if up_names:
                    passes.append(up_names)
                if down_names:
                    passes.append(down_names)
            for names in passes:
                if not names:
                    continue
                acc = _accumulate(student, teacher,
                                  _as_batches(input_ids, batch_size), names, projs,
                                  target_proj=target_proj)
                for n in names:
                    s_mod, t_mod = projs[n]
                    t_w = t_mod.weight.detach().float().cpu()
                    side = "right" if n in down_names else "left"
                    if side == "right" and t_w.shape[0] > t_w.shape[1]:
                        side = "left"
                    if side == "left" and t_w.shape[1] > t_w.shape[0]:
                        side = "right"
                    Wp = _deployed_weight(s_mod).float().cpu()
                    W, st = _solve_from_stats(acc[n]["G"], acc[n]["B"], acc[n]["sst"],
                                              acc[n]["n"], Wp, ridge_lambda)
                    extra = set_deployed_weight(
                        s_mod, W, t_w, side=side, perp=perp,
                        lora_rank=int(getattr(s_mod, "lora_rank", 0) or 0))
                    # Actual deployed weight after factorization / truncation.
                    We = _deployed_weight(s_mod).detach().float().cpu()
                    resid = (acc[n]["sst"] - 2.0 * float((We * acc[n]["B"].T).sum())
                             + float(((We @ acc[n]["G"]) * We).sum()))
                    rel_deployed = (max(resid, 0.0) / max(acc[n]["sst"], 1e-30)) ** 0.5
                    row = {"layer": g["layer"], "base": g["base"], "proj": n,
                           "side": side, **st, "rel_err_deployed": rel_deployed, **extra}
                    rows.append(row)
                    if log is not None:
                        log(f"[ridge] L{g['layer']} {n:9s} rel "
                            f"{st['rel_err_before']:.4f} -> {rel_deployed:.4f} "
                            f"(fit={st['rel_err_after']:.4f}, n={st['n_tokens']}, "
                            f"z_perp={extra.get('z_perp_norm', 0):.3f})")
    finally:
        student.train(was_training)
    valid = [r for r in rows if r["rel_err_before"] > 0]
    return {
        "rows": rows,
        "n_groups": len(groups),
        "n_tokens": max((r["n_tokens"] for r in rows), default=0),
        "perp": perp,
        "joint": bool(joint),
        "mean_rel_before": sum(r["rel_err_before"] for r in valid) / max(len(valid), 1),
        "mean_rel_after": sum(r["rel_err_after"] for r in valid) / max(len(valid), 1),
        "mean_rel_deployed": sum(r.get("rel_err_deployed", r["rel_err_after"])
                                 for r in valid) / max(len(valid), 1),
    }


def pack_calib_ids(ids, seq_len: int = 512, num_seqs: int = 128,
                   eos_id: Optional[int] = None) -> torch.Tensor:
    """Flatten token ids and cut into fixed-length calibration rows."""
    if isinstance(ids, torch.Tensor):
        if ids.dim() == 2 and ids.shape[1] >= seq_len and num_seqs > 0:
            return ids[:num_seqs, :seq_len].to(torch.long).contiguous()
        seqs = [ids.reshape(-1).tolist()]
    elif isinstance(ids, (list, tuple)):
        if ids and not isinstance(ids[0], (list, tuple, torch.Tensor)):
            seqs = [list(ids)]
        else:
            seqs = []
            for x in ids:
                if isinstance(x, torch.Tensor):
                    seqs.append(x.reshape(-1).tolist())
                else:
                    seqs.append(list(x))
    else:
        seqs = [list(ids)]
    flat: List[int] = []
    for s in seqs:
        flat.extend(int(x) for x in s)
        if eos_id is not None:
            flat.append(int(eos_id))
    n = len(flat) // seq_len
    if num_seqs and num_seqs > 0:
        n = min(n, num_seqs)
    if n <= 0:
        raise ValueError(f"calibration data has {len(flat)} tokens < seq_len={seq_len}")
    return torch.tensor(flat[:n * seq_len], dtype=torch.long).view(n, seq_len)


def load_calib_ids(path: str, tokenizer=None, seq_len: int = 512,
                   num_seqs: int = 128) -> torch.Tensor:
    """Load calibration token ids from .pt/.pth/.npy/.jsonl/.json/.txt."""
    ext = os.path.splitext(str(path))[1].lower()
    ids = None
    if ext in (".pt", ".pth", ".bin"):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            for k in ("input_ids", "ids", "tokens", "calib_ids"):
                if k in obj:
                    ids = obj[k]
                    break
            if ids is None:
                raise ValueError(f"{path}: dict has no input_ids/ids/tokens")
        else:
            ids = obj
    elif ext == ".npy":
        import numpy as np
        ids = torch.from_numpy(np.load(path))
    elif ext in (".jsonl", ".json", ".txt"):
        if ext == ".jsonl":
            items = [json.loads(line) for line in open(path) if line.strip()]
        elif ext == ".json":
            obj = json.load(open(path))
            items = obj if isinstance(obj, list) else [obj]
        else:
            items = [{"text": line.strip()} for line in open(path) if line.strip()]
        seqs = []
        for it in items:
            if isinstance(it, dict) and "input_ids" in it:
                seqs.append(it["input_ids"])
            else:
                text = it.get("text", "") if isinstance(it, dict) else str(it)
                if tokenizer is None:
                    raise ValueError(f"{path}: text calibration needs a tokenizer")
                seqs.append(tokenizer(text, add_special_tokens=False).input_ids)
        ids = seqs
    else:
        raise ValueError(f"unsupported calibration file: {path}")
    eos = None
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
    return pack_calib_ids(ids, seq_len=seq_len, num_seqs=num_seqs, eos_id=eos)
