#!/usr/bin/env python
"""Trainable-coordinate budget for TWYD / LRC realizations.

All counts are per MLP projection (gate/up/down are symmetric) and then x3xL.

  plain LRC     W = T Z_col^T,  T frozen (d_ff x H), Z_col (r x H)
                trainable = r*H
  dense         W (d_ff x r) fully trainable
                trainable = r*d_ff
  CORE full     W = T Z_col^T + U_perp Z_perp^T, col frozen
                trainable = r*(d_ff-H)
  CORE-LR(k)    Z_perp = A B^T, A (r x k), B ((d_ff-H) x k)
                trainable = k*(r + (d_ff-H)), update rank <= k
  dense-LoRA(k) W = W_col + A B^T, A (d_ff x k), B (r x k)
                trainable = k*(d_ff + r), update rank <= k

CORE-LR is the TWYD-native efficient variant: every trainable coordinate lives
in the stranded complement, and the merged weight is still one dense d_ff x r
matrix (zero inference cost). Dense-LoRA is a PEFT baseline: its update rank is
only k, so it explores a smaller reachable set than plain LRC when k < r.

Usage:
  python scripts/param_budget.py --H 3072 --dff 8192 --r 1536 --layers 28 --k 128
  python scripts/param_budget.py --H 3072 --dff 8192 --r 1024 --layers 28 --k 64,128,256,512
"""
import argparse
import json


def budget(H: int, dff: int, r: int, layers: int, k: int):
    per = {
        "plain_lrc": r * H,
        "dense": r * dff,
        "core_full": r * (dff - H),
        "core_lr": k * (r + (dff - H)),
        "dense_lora": k * (r + dff),
    }
    total = {name: 3 * layers * v for name, v in per.items()}
    dense_total = total["dense"]
    rows = []
    for name, n in total.items():
        rows.append({
            "name": name,
            "per_proj_M": n / 3 / layers / 1e6,
            "total_M": n / 1e6,
            "vs_dense": n / dense_total,
            "vs_plain": n / total["plain_lrc"],
            "u_vs_deployed": n / dense_total,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=3072, help="teacher hidden")
    ap.add_argument("--dff", type=int, default=8192, help="teacher intermediate")
    ap.add_argument("--r", type=int, default=1536, help="student hidden")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--k", default="64,128,256,512,1024", help="comma-separated adapter ranks")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    max_k = min(args.r, args.dff - args.H)
    out = {"setting": vars(args), "max_k": max_k, "frontier": []}
    for k in [int(x) for x in str(args.k).split(",") if x.strip()]:
        if k > max_k:
            print(f"[warn] k={k} > min(r, d_ff-H)={max_k}: the A B^T factorization "
                  f"is redundant and no longer cheaper than CORE/dense; skipping.")
            continue
        if k == max_k:
            print(f"[warn] k={k} == min(r, d_ff-H): full-rank factorization is redundant; "
                  f"prefer CORE full (fewer params).")
        rows = budget(args.H, args.dff, args.r, args.layers, k)
        out["frontier"].append({"k": k, "rows": rows})

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"teacher H={args.H} d_ff={args.dff} r={args.r} layers={args.layers} "
          f"(deployed per MLP = r*d_ff = {args.r*args.dff/1e6:.2f}M, x3L = "
          f"{3*args.layers*args.r*args.dff/1e9:.3f}B)\n")
    header = f"{'variant':<14}{'per-proj M':>12}{'total M':>12}{'vs dense':>10}{'vs plain':>10}"
    for item in out["frontier"]:
        print(f"--- k={item['k']} ---")
        print(header)
        for r in item["rows"]:
            print(f"{r['name']:<14}{r['per_proj_M']:>12.2f}{r['total_M']:>12.1f}"
                  f"{r['vs_dense']:>10.2%}{r['vs_plain']:>10.2%}")
        print()


if __name__ == "__main__":
    main()
