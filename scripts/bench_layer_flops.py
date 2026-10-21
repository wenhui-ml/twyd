"""Per-layer cost of the TWYD branches on the real Llama-3.2-3B -> r=1536 shapes.

Answers "why is CORE-LR ~10x slower per step than plain LoRA" without needing a
full training run.  Every number is one training step's forward+backward for a
single MLP projection, measured on whichever device is available.

    python scripts/bench_layer_flops.py            # cpu-safe, gpu if present
    python scripts/bench_layer_flops.py --seq 512
"""
import argparse
import time

import torch
import torch.nn as nn

from twyd.tuner import CoreLinear, CoreLinearLoRA, DenseLinearLoRA

# Llama-3.2-3B teacher -> LRC-1.5B student (official LowRankClone layout).
R = 1536          # student residual width
D_FF = 8192       # inherited MLP width
H = 3072          # teacher residual width -> |col(T)| and |col(T)^perp|
P = D_FF - H      # 5120 complement dims available to CORE


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench(mod, x, steps=8, warmup=3):
    dev = x.device
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad],
                            lr=1e-4)
    for _ in range(warmup):
        out = mod(x)
        out.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = mod(x)
        out.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    _sync(dev)
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--layers", type=int, default=28 * 3,
                    help="projections per model (28 layers x gate/up/down)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = getattr(torch, a.dtype)
    g = torch.Generator(device="cpu").manual_seed(0)
    S = a.seq

    x = torch.randn(1, S, R, generator=g).to(dev, dt)

    T = torch.randn(D_FF, H, generator=g)
    Z_col = torch.randn(R, H, generator=g) * 0.02
    U = torch.randn(D_FF, P, generator=g)
    U = torch.linalg.qr(U).Q.contiguous()   # orthonormal complement
    T, Z_col, U = T.to(dev, dt), Z_col.to(dev, dt), U.to(dev, dt)

    W = (T.float() @ Z_col.float().T).to(dev, dt)

    cases = {}

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(W.clone())

        def forward(self, h):
            return nn.functional.linear(h, self.w)

    cases["dense W (r->d_ff)"] = _Dense()
    cases["dense_lora k=%d" % a.k] = DenseLinearLoRA(
        R, D_FF, lora_rank=a.k, lora_alpha=2 * a.k, base_weight=W.clone())
    cases["core full (merged)"] = CoreLinear(T, Z_col, U,
                                            torch.zeros(R, P), side="left",
                                            with_perp=True)
    cases["core_lr k=%d merged" % a.k] = CoreLinearLoRA(
        T, Z_col, U, lora_rank=a.k, lora_alpha=2 * a.k, side="left",
        freeze_col=True, factorized=False)
    cases["core_lr k=%d factorized" % a.k] = CoreLinearLoRA(
        T, Z_col, U, lora_rank=a.k, lora_alpha=2 * a.k, side="left",
        freeze_col=True, factorized=True)

    # Reference FLOPs per token of the plain deployed matmul.
    base_flop = 2 * R * D_FF * S

    print(f"device={dev} dtype={dt} seq={S} k={a.k} "
          f"(per-projection fwd+bwd, then x{a.layers} projections)")
    print(f"{'case':26s} {'ms/step':>9s} {'x dense':>8s}  {'ms/model':>9s}")
    ref = None
    for name, mod in cases.items():
        mod = mod.to(dev, dt)
        n_tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        t = bench(mod, x)
        if ref is None:
            ref = t
        print(f"{name:26s} {t * 1e3:9.3f} {t / ref:8.2f}x {t * a.layers * 1e3:9.1f}"
              f"  trainable={n_tr / 1e6:.2f}M")
    print(f"\nplain deployed matmul FLOP/token: {base_flop / 1e9:.3f} GFLOP")


if __name__ == "__main__":
    main()