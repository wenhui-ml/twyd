#!/usr/bin/env python
"""Rank layers/projections by how much they actually used the opened complement.

The V1.5 question is *where* full-MLP capacity should go, not how to factor it.
Given a width-r base and a full-MLP-trained checkpoint of the same shape, this
script measures, per layer and projection:

  * ``delta_frob``  ||W_trained - W_base||_F                    (how much moved)
  * ``rel_delta``   delta_frob / ||W_base||_F                   (scale-free)
  * ``perp_energy`` sqrt(||dW||^2 - ||T^T dW||^2)               (energy outside col(T))
  * ``perp_frac``   perp_energy / delta_frob                    (share using the complement)

and emits ready-to-use ``LORA_LAYERS=...`` plans for the top-k layers, so the
next run can unfreeze only the layers that matter (paper Fig. 8b).

Usage:
  python scripts/rank_layers.py \
    --base /home/chenwh/models/LRC-1.5B-Base \
    --trained output/fw-dense-pt20M/v0-*/checkpoint-1220 \
    --teacher /home/chenwh/models/Llama-3.2-3B-Instruct \
    --device cuda --topk 3,6,12 --json-out output/layer_rank.json
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


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


def _ranges(layers):
    """[0,1,2,4,7] -> '0-2,4,7' (parse_layer_range accepts both forms)."""
    layers = sorted(set(int(x) for x in layers))
    out, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        out.append(str(layers[i]) if i == j else f"{layers[i]}-{layers[j]}")
        i = j + 1
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="width-r student the run started from")
    ap.add_argument("--trained", required=True, help="full-MLP checkpoint (same shape)")
    ap.add_argument("--teacher", required=True, help="teacher used for the SVD basis")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", default="3,6,12", help="emit LORA_LAYERS plans for these k")
    ap.add_argument("--rank-by", default="perp", choices=["perp", "delta"],
                    help="score used to order layers (default: energy outside col(T))")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from twyd.core_basis import build_core_basis
    from transformers import AutoConfig

    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    scfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    sc = getattr(scfg, "text_config", scfg)
    n_layers = int(getattr(sc, "num_hidden_layers", 0))
    layers = list(range(n_layers)) if args.layers == "all" else [
        int(x) for x in args.layers.split(",") if x.strip()]
    bidx = _load_index(args.base)
    tidx = _load_index(args.teacher)
    ridx = _load_index(args.trained)

    rows = []
    for l in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{l}.mlp.{proj}.weight"
            try:
                Wb = _get(args.base, bidx, name).to(dev, torch.float32)
                Wt = _get(args.trained, ridx, name).to(dev, torch.float32)
                Wtea = _get(args.teacher, tidx, name)
            except Exception as e:  # missing key in one of the dirs
                print(f"[skip] {name}: {e}")
                continue
            dW = Wt - Wb
            if proj in ("gate_proj", "up_proj"):
                T, _U = build_core_basis(Wtea, "left")
                basis = T.to(dev, torch.float32)
                col = basis.T @ dW                  # H x r
                side = "left"
            else:
                R, _V = build_core_basis(Wtea, "right")
                basis = R.to(dev, torch.float32)
                col = dW @ basis.T                  # r x H
                side = "right"
            d2 = float((dW * dW).sum())
            c2 = float((col * col).sum())
            perp2 = max(d2 - c2, 0.0)
            row = {
                "layer": l, "proj": proj, "side": side,
                "delta_frob": d2 ** 0.5,
                "rel_delta": (d2 ** 0.5) / max(float((Wb * Wb).sum()) ** 0.5, 1e-12),
                "col_energy": c2 ** 0.5,
                "perp_energy": perp2 ** 0.5,
                "perp_frac": (perp2 ** 0.5) / max(d2 ** 0.5, 1e-12),
            }
            rows.append(row)
            print(f"L{l:02d} {proj:9s} ||dW||={row['delta_frob']:.4f} "
                  f"rel={row['rel_delta']:.4f} perp={row['perp_energy']:.4f} "
                  f"({100 * row['perp_frac']:.1f}% of dW)")
            del Wb, Wt, dW, col, basis

    if not rows:
        raise SystemExit("no comparable projections found; are base/trained the same shape?")

    by_layer = {}
    for r in rows:
        d = by_layer.setdefault(r["layer"], {"layer": r["layer"], "delta": 0.0, "perp": 0.0})
        d["delta"] += r["delta_frob"] ** 2
        d["perp"] += r["perp_energy"] ** 2
    for d in by_layer.values():
        d["delta"] = d["delta"] ** 0.5
        d["perp"] = d["perp"] ** 0.5
    ranking = sorted(by_layer.values(), key=lambda d: d[args.rank_by], reverse=True)

    total_delta = sum(d["delta"] ** 2 for d in ranking) ** 0.5 or 1.0
    total_perp = sum(d["perp"] ** 2 for d in ranking) ** 0.5 or 1.0
    base_tot = total_perp if args.rank_by == "perp" else total_delta
    plans = {}
    for k in [int(x) for x in str(args.topk).split(",") if x.strip()]:
        chosen = [d["layer"] for d in ranking[:k]]
        plans[f"top{k}"] = _ranges(chosen)
        cum = sum(by_layer[l][args.rank_by] ** 2 for l in chosen) ** 0.5
        print(f"[plan] top{k:<3d} LORA_LAYERS={plans[f'top{k}']:<24s} "
              f"captures {100 * cum / base_tot:.1f}% of {args.rank_by} energy")

    agg = {"rank_by": args.rank_by, "total_delta": total_delta, "total_perp": total_perp,
           "layer_order": [d["layer"] for d in ranking], "plans": plans}
    print("\n[rank] layer order:", agg["layer_order"])
    print("[aggregate]", json.dumps({k: v for k, v in agg.items() if k != "layer_order"}, indent=2))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"config": vars(args), "rows": rows,
                       "layers": ranking, "aggregate": agg}, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
