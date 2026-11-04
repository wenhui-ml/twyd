#!/usr/bin/env python
"""Audit an arm's *weights* against its claims — no GPU, no training needed.

Three things can silently invalidate every number we report, and all three are
visible in the exported checkpoint:

  1. **Coverage.**  Did the arm change exactly the modules it claims and nothing
     else?  (`LORA_LAYERS` / `LORA_MODULES` freeze correctness.)
  2. **Reachability signature.**  For an MLP weight `W` (d_ff x r) and the
     teacher's weight `T` (d_ff x H), the energy fraction of `dW = W_arm - W_base`
     inside `col(T)` is directly measurable:

         plain LRC     : dW = T dZ^T        -> energy inside col(T)  ~ 1
         CORE-LR       : dW = U_perp dZ^T   -> energy inside col(T)  ~ 0
         P6 (col ctrl) : dW = T dZ^T        -> energy inside col(T)  ~ 1
         dense         : unconstrained      -> energy inside col(T)  ~ H/d_ff

  3. **It actually trained.**  Non-trivial ||dW||, and the number of "large"
     singular values of dW (should be <= k for the factorised arms).

Usage:
  python scripts/audit_arm_weights.py \
      --teacher /home/chenwh/models/Llama-3.2-3B-Instruct \
      --base    /home/chenwh/models/LRC-1.5B-Base \
      --arm     output/product/corelr-k256/<run>/checkpoint-2000 \
      --expect  "0-2,11-13,25-27:gate,up" --expect-inside 0.0 --k 256
"""
import argparse
import os
import sys

import torch
from safetensors import safe_open

MLP = ("gate_proj", "up_proj", "down_proj")
TOL = 1e-6


class _Shards:
    """Read-only view over one or more safetensors shards (lazy, no full load)."""

    def __init__(self, files):
        self.readers = [safe_open(f, "pt") for f in files]
        self.keymap = {}
        for r in self.readers:
            for k in r.keys():
                self.keymap[k] = r

    def keys(self):
        return self.keymap.keys()

    def get_tensor(self, k):
        return self.keymap[k].get_tensor(k)


def _load(path):
    if os.path.isdir(path):
        single = os.path.join(path, "model.safetensors")
        if os.path.isfile(single):
            return safe_open(single, "pt"), single
        import glob
        shards = sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors")))
        if shards:
            return _Shards(shards), f"{path} [{len(shards)} shards]"
        sys.exit(f"!! no model.safetensors or shards under {path}")
    return safe_open(path, "pt"), path


def _layers(spec):
    """'0-2,11-13,25-27' -> {0,1,2,11,12,13,25,26,27}; 'all' -> None."""
    if spec in (None, "", "all"):
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _inside_energy(dW, T, Qcache=None, key=None):
    """||Q^T dW||_F^2 / ||dW||_F^2 with Q an orthonormal basis of col(T)."""
    if Qcache is not None and key in Qcache:
        Q = Qcache[key]
    else:
        Q, _ = torch.linalg.qr(T.float())
        if Qcache is not None:
            Qcache[key] = Q
    proj = Q.T @ dW.float()
    return (proj.pow(2).sum() / dW.float().pow(2).sum()).item()


def _rank_stats(dW, k=None):
    """(stable_rank, #sv above 1e-3*max, energy fraction in the top-k)"""
    s = torch.linalg.svdvals(dW.float())
    tot = s.pow(2).sum()
    if tot <= 0:
        return 0.0, 0, 0.0
    sr = (tot / s.max().pow(2)).item()
    n_big = int((s > 1e-3 * s.max()).sum())
    topk = (s[:k].pow(2).sum() / tot).item() if k else float("nan")
    return sr, n_big, topk, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--expect", default="", help="'L:S,M' e.g. '0-2,11-13,25-27:gate,up' or 'all:all'")
    ap.add_argument("--expect-inside", type=float, default=None)
    ap.add_argument("--tol-inside", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max-report", type=int, default=10)
    ap.add_argument("--rel-tol", type=float, default=0.01,
                    help="relative ||dW||/||W_base|| above which a tensor counts as trained "
                         "(bf16 export rounding is ~4e-3)")
    args = ap.parse_args()

    base_r, base_p = _load(args.base)
    arm_r, arm_p = _load(args.arm)
    tea_r, tea_p = _load(args.teacher)
    print(f"base  = {base_p}")
    print(f"arm   = {arm_p}")
    print(f"teach = {tea_p}")

    changed, rows = [], {}
    for i in range(64):
        any_layer = False
        for m in MLP:
            key = f"model.layers.{i}.mlp.{m}.weight"
            if key not in base_r.keys() or key not in arm_r.keys():
                continue
            any_layer = True
            Wb = base_r.get_tensor(key).float()
            dW = arm_r.get_tensor(key).float() - Wb
            n = dW.norm().item()
            rows[(i, m)] = (n, n / max(Wb.norm().item(), 1e-9), dW)
            if (n / max(Wb.norm().item(), 1e-9)) > args.rel_tol:
                changed.append((i, m))
        if not any_layer:
            break

    print(f"\n== 1. coverage: {len(changed)} MLP tensors changed")
    if args.expect:
        ls, ms = args.expect.split(":")
        exp_l = _layers(ls)
        exp_m = set(MLP) if ms == "all" else {x.strip() + "_proj" for x in ms.split(",")}
        want = {(i, m) for (i, m) in rows if (exp_l is None or i in exp_l) and m in exp_m}
        got = set(changed)
        extra, missing = got - want, want - got
        print(f"   expected {len(want)}; changed {len(got)}")
        if extra:
            print(f"   !! UNEXPECTED: {sorted(extra)[:8]}{' ...' if len(extra) > 8 else ''}")
        if missing:
            print(f"   !! MISSING: {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
        if not extra and not missing:
            print("   PASS: changed set == expected set")

    print(f"\n== 2/3. delta geometry (first {args.max_report} changed tensors)")
    print(f"   {'layer':>5} {'module':<9} {'|dW|':>9} {'rel':>8} {'in col(T)':>10} {'stable_r':>9} {'#sv':>5} {'topk_e':>9}")
    qc, inside = {}, []
    for (i, m) in changed[: args.max_report]:
        n, rel, dW = rows[(i, m)]
        tk = f"model.layers.{i}.mlp.{m}.weight"
        e = None
        if tk in tea_r.keys():
            T = tea_r.get_tensor(tk)
            if T.shape[0] == dW.shape[0]:      # gate/up: (d_ff x H) vs (d_ff x r)
                e = _inside_energy(dW, T, qc, (i, m))
                inside.append(e)
        sr, nbig, topk, sv = _rank_stats(dW, args.k)
        print(f"   {i:>5} {m:<9} {n:9.4f} {rel:8.4f} "
              f"{(f'{e:.3f}' if e is not None else 'n/a'):>10} {sr:9.2f} {nbig:5d} "
              f"{(f'{topk:.4f}' if args.k else ''):>9}")

    if inside:
        mu = sum(inside) / len(inside)
        print(f"\n   mean energy inside col(T): {mu:.3f}  (n={len(inside)})")
        if args.expect_inside is not None:
            ok = abs(mu - args.expect_inside) <= args.tol_inside
            print(f"   expected {args.expect_inside:.2f} +-{args.tol_inside:.2f} -> "
                  f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
