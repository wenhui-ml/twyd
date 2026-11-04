#!/usr/bin/env python
"""Paired item-level comparison of lm-eval-harness runs (McNemar + recovery fraction).

Comparing "acc 61.2% vs 60.4%" with independent standard errors is hopeless at
lm-eval sample sizes (SE ~0.5 pt on hellaswag, ~1.4 pt on arc_easy). Every arm here
sees the *same* items, so the right test is paired: count the items where exactly
one arm is right (discordant pairs) and run McNemar on them.

The second statistic is the one reviewers ask for: how much of the full-model gain
over the base model does an arm recover?

    recovery = (acc_arm - acc_base) / (acc_full - acc_base)

Usage (paths may be a samples_*.jsonl file or a directory containing them):

    python scripts/eval_mcnemar.py --metric acc_norm \
        base=output/lm_eval/base/ full=output/lm_eval/A525/ sel9=output/lm_eval/B525/
"""
import argparse
import glob
import math
import os
import re
from collections import defaultdict

METRIC_FALLBACKS = ("acc_norm", "acc", "exact_match")

TASK_RE = re.compile(r"samples_([A-Za-z0-9_]+?)_\d{4}-\d{2}-\d{2}")


def _task_of(path):
    m = TASK_RE.search(os.path.basename(path))
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(path)) or "task"


def _load(path):
    """-> {doc_id: score} for one samples file."""
    import json

    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            score = None
            for key in METRIC_FALLBACKS:
                if key in rec and rec[key] is not None:
                    score = float(rec[key])
                    break
            if score is None:
                raise SystemExit(f"{path}: no accuracy field among {METRIC_FALLBACKS}")
            out[rec.get("doc_id", i)] = 1.0 if score >= 0.5 else 0.0
    return out


def _collect(path):
    """-> {task: {doc_id: score}} for a file or a directory of samples files."""
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "samples_*.jsonl"),
                                 recursive=True))
    elif os.path.isfile(path):
        files = [path]
    if not files:
        raise SystemExit(f"no samples_*.jsonl under {path} (did you pass --log_samples?)")
    res = defaultdict(dict)
    for f in files:
        res[_task_of(f)].update(_load(f))
    return res


def _mcnemar(b, c):
    """Two-sided McNemar. Exact binomial for small n, normal approx otherwise."""
    if b + c == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(min(b, c)), int(b + c), 0.5).pvalue)
    except Exception:
        n = b + c
        z = (abs(b - c) - 1) / math.sqrt(n) if n else 0.0
        return math.erfc(abs(z) / math.sqrt(2))


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (mid - half, mid + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=path[,name=path...] (first = reference)")
    ap.add_argument("--metric", default="acc_norm",
                    help="preferred metric; falls back to acc / exact_match per record")
    ap.add_argument("--recovery-of", default=None,
                    help="arm whose gain over the reference sets the denominator "
                         "(default: the second arm)")
    args = ap.parse_args()

    arms = []
    for spec in args.arms:
        for chunk in spec.split(","):
            if "=" not in chunk:
                raise SystemExit(f"bad arm spec {chunk!r}, want name=path")
            name, path = chunk.split("=", 1)
            arms.append((name, _collect(path)))
    if len(arms) < 2:
        raise SystemExit("need at least two arms")
    ref_name, ref = arms[0]
    rec_name = args.recovery_of or arms[1][0]
    acc_of = dict(arms)

    tasks = sorted(set().union(*[set(m) for _, m in arms]))
    print(f"reference = {ref_name}   metric = {args.metric}   arms = "
          f"{', '.join(n for n, _ in arms)}")

    for task in tasks:
        common = set(ref.get(task, {}))
        for _, m in arms:
            common &= set(m.get(task, {}))
        ids = sorted(common)
        if not ids:
            continue
        n = len(ids)
        print(f"\n=== {task}   n={n}")
        acc = {}
        for name, m in arms:
            sc = m.get(task, {})
            k = sum(sc.get(i, 0.0) for i in ids)
            acc[name] = k / n
            lo, hi = _wilson(k, n)
            print(f"  {name:<12s} acc={acc[name]:.4f}  [{lo:.4f},{hi:.4f}]  "
                  f"({int(k)}/{n})")

        print(f"  paired vs {ref_name}:")
        for name, m in arms[1:]:
            sc = m.get(task, {})
            b = sum(1 for i in ids if ref[task].get(i) == 1 and sc.get(i) == 0)
            c = sum(1 for i in ids if ref[task].get(i) == 0 and sc.get(i) == 1)
            diff = [sc.get(i, 0.0) - ref[task].get(i, 0.0) for i in ids]
            mean = sum(diff) / n
            var = sum((x - mean) ** 2 for x in diff) / max(n - 1, 1)
            se = math.sqrt(var / n)
            p = _mcnemar(b, c)
            print(f"    {name:<12s} d={mean:+.4f} [{mean - 1.96 * se:+.4f},"
                  f"{mean + 1.96 * se:+.4f}]  discordant {b}/{c}  McNemar p={p:.2e}")

        if len(arms) >= 3 and rec_name != ref_name:
            denom = acc[rec_name] - acc[ref_name]
            print(f"  recovery of {rec_name} gain over {ref_name}:")
            for name, _ in arms[1:]:
                num = acc[name] - acc[ref_name]
                if abs(denom) < 1e-9:
                    print(f"    {name:<12s} n/a ({rec_name} gain ~ 0)")
                else:
                    print(f"    {name:<12s} {num / denom * 100:+.1f}%")


if __name__ == "__main__":
    main()
