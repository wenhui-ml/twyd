#!/usr/bin/env python
"""Precise status for a TWYD run: watch CE, not the composite loss.

The trainer's loss is `CE + T^2 * KL` (T=40, so the KL term is ~179 while CE is ~2.5).
A 1% CE improvement therefore moves the printed loss by 0.03 and looks like "no learning".
`twyd/loss.py` prints the components every 10 steps:

    [twyd] ce=2.4566 kl_raw=0.000208 klT2=0.33 loss=2.62

This tool pairs those lines with the tqdm step counter, so you get a real CE curve plus
speed / ETA / VRAM / grad_norm / checkpoints, and it flags a stalled log.

  python scripts/watch_training.py                              # newest run
  python scripts/watch_training.py output/formal-b/slice-full    # one arm
  python scripts/watch_training.py --follow 60                   # refresh every 60 s
  python scripts/watch_training.py --eval-hint                    # print held-out eval cmd

`kl_raw` tells you the regime: ~2e-4 right after a teacher slice (readable, lots of
headroom) vs 0.1+ when continuing from the released base (saturated, little headroom).
"""
import argparse
import glob
import json
import os
import re
import time

TQDM_RE = re.compile(r"Train:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
CE_RE = re.compile(r"\[twyd\] ce=([\d.]+) kl_raw=([\d.eE+-]+) klT2=([\d.]+) loss=([\d.]+)")


def human(sec):
    if sec != sec or sec in (float("inf"), float("-inf")):
        return "?"
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def resolve(run):
    run = run.rstrip("/")
    if os.path.exists(os.path.join(run, "logging.jsonl")):
        return run
    for pat in (f"{run}/*/logging.jsonl", f"{run}/*/*/logging.jsonl"):
        hits = sorted(glob.glob(pat), key=os.path.getmtime)
        if hits:
            return os.path.dirname(hits[-1])
    return run


def find_runs(explicit):
    if explicit:
        return [explicit]
    cands = [os.path.dirname(p) for p in glob.glob("output/**/logging.jsonl", recursive=True)]
    return sorted(cands, key=lambda d: os.path.getmtime(os.path.join(d, "logging.jsonl")),
                  reverse=True)[:6]


def load_steps(vdir):
    path = os.path.join(vdir, "logging.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "loss" in r]


def parse_ce_curve(logpath):
    """[(step, ce, kl_raw, loss)] from the interleaved tqdm/print stream."""
    if not os.path.exists(logpath):
        return []
    with open(logpath, "rb") as f:
        raw = f.read().decode("utf-8", "ignore")
    out, cur = [], 0
    for tok in re.split(r"[\r\n]", raw):
        m = TQDM_RE.search(tok)
        if m:
            cur = int(m.group(1))
        m = CE_RE.search(tok)
        if m:
            out.append((cur, float(m.group(1)), float(m.group(2)), float(m.group(4))))
    return out


def ema(vals, alpha=0.15):
    if not vals:
        return float("nan")
    e = vals[0]
    for v in vals[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def slope(points, tail=None):
    pts = points[-tail:] if tail else points
    if len(pts) < 3:
        return float("nan")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def report(arm, tail_n=8, eval_hint=False):
    vdir = resolve(arm)
    log = os.path.normpath(os.path.join(vdir, "..")) + ".log"
    if not os.path.exists(log):
        log = os.path.normpath(os.path.join(vdir, "..", "..")) + ".log"
    steps = load_steps(vdir)
    curve = parse_ce_curve(log)

    print(f"{vdir}")
    if steps:
        last = steps[-1]
        cur, total = (last["global_step/max_steps"].split("/") + ["?"])[:2]
        rate = float(last.get("train_speed(s/it)", float("nan")))
        pct = 100.0 * int(cur) / int(total) if total.isdigit() and int(total) else 0.0
        print(f"  step      {cur}/{total} ({pct:.1f}%)   elapsed {last.get('elapsed_time', '?')}")
        if rate == rate:
            print(f"  speed     {rate:.2f} s/step   ETA {human((int(total) - int(cur)) * rate)}")
        gns = [r["grad_norm"] for r in steps if r.get("grad_norm") is not None]
        if gns:
            s = sorted(gns)
            print(f"  grad_norm median {s[len(s) // 2]:.1f}   max {max(gns):.1f}")
        mem = max((r.get("memory(GiB)") or 0) for r in steps)
        print(f"  peak mem  {mem:.2f} GiB (reserved)")
        print(f"  acc       {last.get('token_acc', 0):.4f}")
        print(f"  loss      {last['loss']:.2f}  <- CE + T^2*KL; read the CE line below instead")
    else:
        print("  (no step logged yet -- dataset encoding / model load)")

    if curve:
        ce = [c[1] for c in curve]
        print(f"  CE        last {ce[-1]:.4f}   ema {ema(ce):.4f}   min {min(ce):.4f}   n={len(ce)}")
        print(f"  CE slope  per 1k steps: all {slope(curve) * 1000:+.4f}   "
              f"last{tail_n} {slope(curve, tail_n) * 1000:+.4f}   (negative = learning)")
        kr = curve[-1][2]
        regime = ("teacher-slice regime (readable, lots of headroom)" if kr < 1e-3
                  else "continued-from-base regime (saturated, little headroom)")
        print(f"  kl_raw    {kr:.6f}  -> {regime}")
        for st, c, k, lo in curve[-min(6, len(curve)):]:
            print(f"    step {st:>6}  ce {c:.4f}  kl_raw {k:.6f}  loss {lo:.2f}")
    else:
        print("  CE        (no [twyd] ce= lines: this run started before the component")
        print("             logging patch of 2026-10-23 09:30, or the log path is wrong)")

    ck = sorted(glob.glob(os.path.join(vdir, "checkpoint-*")))
    if ck:
        print(f"  ckpt      {', '.join(os.path.basename(c) for c in ck[-3:])}")
        if eval_hint:
            print("  held-out CE for the newest ckpt (needs ~7 GiB free VRAM):")
            print(f"    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ce.py "
                  f"--data data/fineweb_edu_holdout.jsonl")
            print(f"      --models /home/chenwh/models/LRC-1.5B-Base {ck[-1]} "
                  f"--seq-len 512 --max-docs 600")

    logf = os.path.join(vdir, "logging.jsonl")
    if os.path.exists(logf) and steps:
        idle = time.time() - os.path.getmtime(logf)
        rate = float(steps[-1].get("train_speed(s/it)", 30))
        if idle > max(300, 6 * rate):
            print(f"  !! no new log for {human(idle)} -- stalled/crashed? check {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--follow", type=int, default=0, metavar="SEC")
    ap.add_argument("--points", type=int, default=8, help="window for the tail CE slope")
    ap.add_argument("--eval-hint", action="store_true")
    args = ap.parse_args()

    while True:
        runs = find_runs(args.run)
        if not runs:
            print("no run dir with logging.jsonl under output/")
            return
        for i, r in enumerate(runs):
            if i:
                print()
            report(r, args.points, args.eval_hint if i == 0 else False)
        if not args.follow:
            return
        print(f"\n(refresh {args.follow}s, Ctrl-C to stop)")
        time.sleep(args.follow)
        print("\033[2J\033[H", end="")


if __name__ == "__main__":
    main()

