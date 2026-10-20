#!/usr/bin/env python
"""Measure the effective rank of the learned complement (TWYD-LR design tool).

For each MLP projection, decompose the deployed width-r student weight as

    W_s = W_col + dW
    W_col = T (T^T W_s)          # gate/up: projection onto col(T)
    W_col = (W_s R^T) R          # down:    projection onto row(W)
    Z_perp^T = U_perp^T dW       # gate/up
    Z_perp   = dW V_perp^T       # down

and report the singular-value spectrum of Z_perp / dW:
  frob, spectral norm, stable rank, effective rank (exp entropy), top-k energy.

This answers the TWYD-LR question: how many complement directions k are
actually used by a full Dense/CORE run? If top-k energy saturates at k << r,
CORE-LR(k) can recover the gain with k(r + d_ff - H) trainable coordinates.

Usage:
  # measure a trained full Dense/CORE checkpoint against its teacher
  python scripts/measure_complement_rank.py \
      --student /path/to/trained-width-r-model \
      --teacher /path/to/teacher --layers 0,7,14,21,27 --topk 32,64,128,256,512

  # sanity check without HF downloads
  python scripts/measure_complement_rank.py --synthetic
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def spectrum_metrics(Z: torch.Tensor, topk):
    s = torch.linalg.svdvals(Z.detach().float().cpu().reshape(Z.shape[0], -1))
    e = s.pow(2)
    total = e.sum().clamp_min(1e-30)
    p = e / total
    stable = (total.pow(2) / e.pow(2).sum().clamp_min(1e-30)).item()
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    top = {int(k): (e[:k].sum() / total).item() for k in topk}
    return {
        "frob": float(total.sqrt()),
        "spec": float(s[0]) if s.numel() else 0.0,
        "stable_rank": float(stable),
        "erank": float(erank),
        "topk_energy": top,
        "sv": [float(x) for x in s[:min(32, s.numel())]],
    }


def _synthetic():
    """W = T Z_col^T + U Z_perp^T with Z_perp rank k; top-k energy must be ~1."""
    from twyd.core_basis import build_core_basis
    torch.manual_seed(0)
    d_ff, H, r, k = 96, 32, 16, 4
    Wt = torch.randn(d_ff, H)
    T, U = build_core_basis(Wt, "left")
    Zc = torch.randn(r, H)
    A = torch.linalg.qr(torch.randn(r, k)).Q
    B = torch.linalg.qr(torch.randn(d_ff - H, k)).Q
    Ws = T @ Zc.T + U @ (A @ B.T).T
    dW = Ws - T @ (T.T @ Ws)
    Z = U.T @ dW
    m = spectrum_metrics(Z, [k, 2 * k, 8 * k])
    print(json.dumps(m, indent=2))
    assert m["topk_energy"][k] > 0.999, m
    assert abs(m["stable_rank"] - k) < 0.2, m


def _load_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        return json.load(open(idx))["weight_map"]
    return None


def _get(model_dir, index, name):
    from safetensors import safe_open
    if index is None:
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    with safe_open(os.path.join(model_dir, index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None, help="trained width-r HF model dir")
    ap.add_argument("--teacher", default=None, help="teacher HF dir (same teacher used for LRC)")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--topk", default="32,64,128,256,512")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    topk = [int(x) for x in args.topk.split(",") if x.strip()]
    if args.synthetic:
        _synthetic()
        return
    if not args.student or not args.teacher:
        raise SystemExit("--student and --teacher are required (or use --synthetic)")

    from twyd.core_basis import build_core_basis
    from transformers import AutoConfig

    scfg = AutoConfig.from_pretrained(args.student, trust_remote_code=True)
    tcfg = AutoConfig.from_pretrained(args.teacher, trust_remote_code=True)
    sc = getattr(scfg, "text_config", scfg)
    tc = getattr(tcfg, "text_config", tcfg)
    n_layers = int(getattr(sc, "num_hidden_layers", 0))
    sidx, tidx = _load_index(args.student), _load_index(args.teacher)
    layers = list(range(n_layers)) if args.layers == "all" else [
        int(x) for x in args.layers.split(",") if x.strip()]

    rows = []
    for l in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{l}.mlp.{proj}.weight"
            Ws = _get(args.student, sidx, name).float()
            Wt = _get(args.teacher, tidx, name).float()
            if proj in ("gate_proj", "up_proj"):
                T, U = build_core_basis(Wt, "left")
                Wcol = T @ (T.T @ Ws)
                dW = Ws - Wcol
                Z = U.T @ dW
            else:
                R, V = build_core_basis(Wt, "right")
                Wcol = (Ws @ R.T) @ R
                dW = Ws - Wcol
                Z = dW @ V.T
            m = spectrum_metrics(Z, topk)
            m.update({"layer": l, "proj": proj, "shape": list(Z.shape)})
            rows.append(m)
            print(f"L{l:02d} {proj:9s} shape={tuple(Z.shape)} frob={m['frob']:.4f} "
                  f"stable={m['stable_rank']:.1f} erank={m['erank']:.1f} "
                  f"top{topk[-1]}={m['topk_energy'][topk[-1]]:.3f}")

    agg = {}
    for k in topk:
        vals = [x["topk_energy"][k] for x in rows]
        agg[f"top{k}_energy_mean"] = sum(vals) / max(len(vals), 1)
    agg["stable_rank_mean"] = sum(x["stable_rank"] for x in rows) / max(len(rows), 1)
    agg["erank_mean"] = sum(x["erank"] for x in rows) / max(len(rows), 1)
    print("\n[aggregate]", json.dumps(agg, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"config": vars(args), "rows": rows, "aggregate": agg}, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

    print("[synthetic] OK: top-k energy ~1 and stable_rank ~k")
