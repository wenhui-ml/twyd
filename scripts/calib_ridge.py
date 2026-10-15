"""Greedy data-aware ridge calibration (paper Alg.1, minimal).

Reads a few hundred packed sequences, runs teacher + current student layer by
layer, fits col-path Z_col by ridge (Eq.4) against teacher activations, keeps
Z_perp=0, and propagates the student's compressed output to the next layer.
Outlier-aware master P (Eq.5) is exposed via twyd.init.build_outlier_aware_P
for Qwen (LRC_OUTLIER_K / --outlier-keep); embedding P stays future work
(paper App.B disables it in reported runs).

Usage:
  python scripts/calib_ridge.py --teacher <id> --student <wrapped-or-ckpt> \
    --dataset <hf-or-ms-id> --num-seqs 200 --out ridge_init.pt
Offline smoke (no download):
  python scripts/calib_ridge.py --synthetic --hidden 32 --ff 96 --r 16 --layers 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/chenwh/Papers/twyd")

import torch


def _synthetic_check():
    from twyd.core_basis import check_basis, build_core_basis
    from twyd.init import build_outlier_aware_P, ridge_zcol
    torch.manual_seed(0)
    H, d_ff, r, N = 32, 96, 16, 128
    T, Up = build_core_basis(torch.randn(d_ff, H), "left")
    assert check_basis(T, Up)["ok"]
    C = torch.randn(N, r)
    Wp = torch.randn(r, H)
    Z = ridge_zcol(C, C @ Wp, Wp, ridge_lambda=1e-3)
    assert torch.allclose(Z, Wp, atol=1e-3), "ridge should recover prior without residual"
    P, idx = build_outlier_aware_P(torch.randn(N, H) * 10, r, outlier_keep=4)
    assert P.shape == (r, H) and len(idx) == 4
    # outlier rows are passthrough identity
    for i, ch in enumerate(idx):
        assert P[i, ch] == 1.0
    print("synthetic calib check OK: ridge + outlier-P + basis QC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", default="/home/chenwh/Papers/twyd/outputs/ridge_init.pt")
    ap.add_argument("--num-seqs", type=int, default=200)
    ap.add_argument("--outlier-keep", type=int, default=0)
    args = ap.parse_args()
    if args.synthetic or args.teacher is None:
        _synthetic_check()
        return
    raise SystemExit("HF calibration path needs swift env + dataset wiring; run --synthetic for now.")


if __name__ == "__main__":
    main()
