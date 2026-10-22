#!/usr/bin/env python
"""Compact status dashboard for a TWYD training run (works on any ms-swift run dir).

Reads the structured `logging.jsonl` that swift writes every `logging_steps`, so it
does not have to parse the tqdm bars.

  python scripts/watch_training.py                                   # newest run under output/
  python scripts/watch_training.py output/formal-b/full-28L/all     # a specific run dir
  python scripts/watch_training.py --follow 60                       # refresh every 60 s
  python scripts/watch_training.py --tail 5                          # also show raw log lines

What is "healthy" (established on this project):
  * s/step flat (±5%) after the first ~20 steps
  * token_acc rising, then flat near the data's ceiling (~0.474 on FineWeb-Edu)
  * grad_norm staying in the same decade (median ~60-80 for full-model, ~5-20 for 1-2 layers)
  * peak memory flat, not creeping
  * loss: only the *relative* slope is meaningful across configs -- `loss = CE + T^2 * KL`
    with T=40, so the value is dominated by the KL term and is not comparable across
    sequence lengths.
"""
import argparse
import glob
import json
import os
import time


def find_runs(explicit):
    if explicit:
        return [explicit.rstrip("/")]
    cands = [os.path.dirname(p) for p in glob.glob("output/**/logging.jsonl", recursive=True)]
    return sorted(cands, key=lambda d: os.path.getmtime(os.path.join(d, "logging.jsonl")),
                  reverse=True)[:6]


def resolve(run):
    """Accept an arm dir, a tag dir or a version dir; return the version dir with the log."""
    run = run.rstrip("/")
    if os.path.exists(os.path.join(run, "logging.jsonl")):
        return run
    for pat in (f"{run}/*/logging.jsonl", f"{run}/*/*/logging.jsonl"):
        hits = sorted(glob.glob(pat), key=os.path.getmtime)
        if hits:
            return os.path.dirname(hits[-1])
    return run


def load_steps(run):
    path = os.path.join(run, "logging.jsonl")
    if not os.path.exists(path):
        return [], []
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "loss" in r], rows


def human(sec):
    if sec != sec or sec in (float("inf"), float("-inf")):
        return "?"
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def report(run, tail=0):
    run = resolve(run)
    steps, rows = load_steps(run)
    name = run
    if not steps:
        print(f"{name}\n  no step logged yet (still loading / encoding the dataset)")
        return None
    last = steps[-1]
    cur, total = (last["global_step/max_steps"].split("/") + ["?"])[:2]
    cur_i = int(cur)
    tot_i = int(total) if total.isdigit() else 0

    # instantaneous rate from the last two logged points
    rate = float("nan")
    if len(steps) >= 2:
        rate = float(last.get("train_speed(s/it)", float("nan")))
    loss = [r["loss"] for r in steps]
    acc = [r.get("token_acc", float("nan")) for r in steps]
    gn = [r.get("grad_norm") for r in steps if r.get("grad_norm") is not None]
    mem = max((r.get("memory(GiB)", 0) or 0) for r in steps)

    print(f"{name}")
    pct = 100.0 * cur_i / tot_i if tot_i else 0.0
    print(f"  step      {cur}/{total}  ({pct:.1f}%)   elapsed {last.get('elapsed_time', '?')}")
    if tot_i and cur_i and rate == rate:
        print(f"  speed     {rate:.2f} s/step    ETA {human((tot_i - cur_i) * rate)}"
              f"   step {cur_i}/{tot_i}")
    print(f"  loss      last {loss[-1]:8.2f}   first {loss[0]:8.2f}   min {min(loss):8.2f}"
          f"   mean5 {sum(loss[-5:]) / len(loss[-5:]):8.2f}   d={100 * (loss[-1] / loss[0] - 1):+.1f}%")
    if acc and acc[-1] == acc[-1]:
        print(f"  token_acc last {acc[-1]:.4f}   max {max(a for a in acc if a == a):.4f}")
    if gn:
        gn_sorted = sorted(gn)
        print(f"  grad_norm median {gn_sorted[len(gn_sorted) // 2]:.1f}   max {max(gn):.1f}")
    print(f"  peak mem  {mem:.2f} GiB")
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoint-*")) +
                   glob.glob(os.path.join(run, "*", "v*", "checkpoint-*")))
    if ckpts:
        print("  ckpt      " + ", ".join(os.path.basename(c) for c in ckpts[-4:]))

    # staleness / crash detection: compare the last log line's mtime with its own rate
    path = os.path.join(run, "logging.jsonl")
    idle = time.time() - os.path.getmtime(path)
    if rate == rate and idle > max(300, 6 * rate):
        print(f"  !! no new log line for {human(idle)} -- stalled or crashed? "
              f"(check {run.replace('/logging.jsonl', '')}.log)")
    if tail:
        log = os.path.join(run, "..", "..", ".log")
        log = os.path.normpath(os.path.join(run, "..")) + ".log"
        if os.path.exists(log):
            print("  --- last log lines")
            with open(log, "rb") as f:
                f.seek(max(0, os.path.getsize(log) - 4000))
                chunk = f.read().decode("utf-8", "ignore").replace("\r", "\n")
            for line in [x for x in chunk.split("\n") if x.strip()][-tail:]:
                print("   ", line[:160])
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None, help="run dir (default: newest under output/)")
    ap.add_argument("--follow", type=int, default=0, metavar="SEC", help="refresh every SEC seconds")
    ap.add_argument("--tail", type=int, default=0, help="also print the last N raw log lines")
    args = ap.parse_args()

    while True:
        runs = find_runs(args.run)
        if not runs:
            print("no run dir with logging.jsonl found under output/")
            return
        for i, r in enumerate(runs):
            if i:
                print()
            report(r, tail=args.tail if i == 0 else 0)
        if not args.follow:
            return
        print(f"\n(refreshing every {args.follow}s -- Ctrl-C to stop)")
        time.sleep(args.follow)
        print("\033[2J\033[H", end="")


if __name__ == "__main__":
    main()
