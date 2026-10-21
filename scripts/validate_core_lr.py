#!/usr/bin/env python
"""Validate CORE-LR mechanics and benchmark merged vs factorized forward.

  python scripts/validate_core_lr.py --synthetic
  python scripts/validate_core_lr.py --benchmark --device cuda

`synthetic` checks init == plain-LRC, factorized forward/backward == merged
dense path, and rank-map resolution.  `benchmark` compares one realistic MLP
projection (default H=2048, d_ff=8192, r=512, k=128, 1938 tokens).
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from twyd.config import module_rank, parse_rank_map
from twyd.tuner import CoreLinearLoRA


def _qr_basis(rows, cols, device="cpu", dtype=torch.float32):
    if rows >= cols:
        q, _ = torch.linalg.qr(torch.randn(rows, cols, device=device, dtype=dtype))
        return q.contiguous()
    # wide matrix: orthonormal rows via QR on the transpose
    q, _ = torch.linalg.qr(torch.randn(cols, rows, device=device, dtype=dtype))
    return q.T.contiguous()


def _make_module(side, H, d_ff, r, k, factorized, device="cpu", dtype=torch.float32):
    bdt = torch.float32
    if side == "left":
        T = _qr_basis(d_ff, H, device=device, dtype=bdt)
        U = _qr_basis(d_ff, d_ff - H, device=device, dtype=bdt)
        Zc = torch.randn(r, H, device=device, dtype=bdt) * 0.02
    else:
        R = _qr_basis(H, d_ff, device=device, dtype=bdt)
        V = _qr_basis(d_ff - H, d_ff, device=device, dtype=bdt)
        Zc = torch.randn(r, H, device=device, dtype=bdt) * 0.02
    return CoreLinearLoRA(T if side == "left" else R, Zc,
                          U if side == "left" else V,
                          lora_rank=k, lora_alpha=16.0, side=side,
                          freeze_col=True, factorized=factorized).to(device=device, dtype=dtype)


def _linear(m, x):
    return F.linear(x, m.merged_weight())


def _check(side, H=24, d_ff=64, r=12, k=4):
    torch.manual_seed(0)
    merged = _make_module(side, H, d_ff, r, k, False)
    fact = _make_module(side, H, d_ff, r, k, True)
    with torch.no_grad():
        fact.z_col.copy_(merged.z_col)
        fact.basis_col.copy_(merged.basis_col)
        fact.basis_perp.copy_(merged.basis_perp)
        fact.refresh_cache()
    wcol = (merged.basis_col @ merged.z_col.T if side == "left"
            else merged.z_col @ merged.basis_col)
    assert torch.allclose(merged.merged_weight(), wcol, atol=1e-6), "init != plain-LRC"
    assert torch.allclose(fact.W_col_cache, wcol, atol=1e-6), "cache != W_col"
    with torch.no_grad():
        merged.lora_A.normal_(std=0.05)
        merged.lora_B.normal_(std=0.05)
        fact.lora_A.copy_(merged.lora_A)
        fact.lora_B.copy_(merged.lora_B)
    in_dim = r if side == "left" else d_ff
    x = torch.randn(3, in_dim)
    y_ref = _linear(merged, x)
    y_fact = fact(x)
    assert torch.allclose(y_ref, y_fact, atol=2e-5), (side, (y_ref - y_fact).abs().max())
    up = torch.randn_like(y_ref)
    merged.zero_grad(set_to_none=True)
    (_linear(merged, x) * up).sum().backward()
    gA, gB = merged.lora_A.grad.clone(), merged.lora_B.grad.clone()
    fact.zero_grad(set_to_none=True)
    (fact(x) * up).sum().backward()
    assert torch.allclose(gA, fact.lora_A.grad, atol=2e-5, rtol=2e-4), side + " dA"
    assert torch.allclose(gB, fact.lora_B.grad, atol=2e-5, rtol=2e-4), side + " dB"
    print(f"[synthetic] {side}: parity OK (out={tuple(y_ref.shape)})")


def synthetic():
    _check("left")
    _check("right")
    rm = parse_rank_map(json.dumps({"layers.0.gate_proj": 128, "down_proj": 64}))
    assert module_rank(rm, "model.layers.0.mlp.gate_proj", 256) == 128
    assert module_rank(rm, "model.layers.3.mlp.down_proj", 256) == 64
    assert module_rank(rm, "model.layers.3.mlp.up_proj", 256) == 256
    print("[synthetic] rank map OK")
    print("[synthetic] PASS")


def _time(m, x, iters=10, warmup=3):
    for _ in range(warmup):
        m.zero_grad(set_to_none=True)
        y = m(x)
        y.sum().backward()
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        m.zero_grad(set_to_none=True)
        y = m(x)
        y.sum().backward()
    if x.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**30 if x.is_cuda else 0.0
    return dt, peak


def benchmark(args):
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"[bench] side={args.side} H={args.H} d_ff={args.d_ff} r={args.r} "
          f"k={args.k} tokens={args.tokens} device={dev} dtype={dtype}")
    merged = _make_module(args.side, args.H, args.d_ff, args.r, args.k, False, dev, dtype)
    fact = _make_module(args.side, args.H, args.d_ff, args.r, args.k, True, dev, dtype)
    with torch.no_grad():
        fact.z_col.copy_(merged.z_col)
        fact.basis_col.copy_(merged.basis_col)
        fact.basis_perp.copy_(merged.basis_perp)
        fact.refresh_cache()
        merged.lora_A.normal_(std=0.02)
        merged.lora_B.normal_(std=0.02)
        fact.lora_A.copy_(merged.lora_A)
        fact.lora_B.copy_(merged.lora_B)
    in_dim = args.r if args.side == "left" else args.d_ff
    x = torch.randn(args.tokens, in_dim, device=dev, dtype=dtype)
    tm, pm = _time(merged, x, args.iters)
    tf, pf = _time(fact, x, args.iters)
    print(f"[bench] merged      {tm*1e3:8.2f} ms  peak={pm:6.2f} GiB")
    print(f"[bench] factorized  {tf*1e3:8.2f} ms  peak={pf:6.2f} GiB  speedup={tm/tf:.2f}x")
    print("[bench] NOTE: factorized cuts the W-construction term, not the "
          "base W_col GEMM; it requires freeze_col=True.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--d-ff", dest="d_ff", type=int, default=8192)
    ap.add_argument("--r", type=int, default=512)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=1938)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    if not args.synthetic and not args.benchmark:
        args.synthetic = True
    if args.synthetic:
        synthetic()
    if args.benchmark:
        benchmark(args)


if __name__ == "__main__":
    main()
