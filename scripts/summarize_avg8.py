#!/usr/bin/env python
"""Collapse lm-eval samples into the paper's task list and report Avg8 + paired stats.

lm-eval expands the `mmlu` group into 57 subtask files, which makes the raw
eval_mcnemar output unreadable. This aggregates mmlu_* -> mmlu, scores every arm on
the same items, and prints:
  * per-task accuracy (acc_norm when the task has it, else acc)
  * Avg8 = mean over the 8 paper tasks present (LogiQA is unavailable)
  * pooled McNemar for each arm vs the reference, and for the last arm vs the second

    python /tmp/summarize_avg8.py output/lm_eval/paper_pt_base \
        output/lm_eval/paper_pt_A output/lm_eval/paper_pt_B
"""
import glob
import json
import os
import sys

from scipy.stats import binomtest


def task_of(path):
    base = os.path.basename(path)
    name = base.split("samples_")[1].rsplit("_20", 1)[0]
    return "mmlu" if name.startswith("mmlu_") else name


def subkey(path):
    """Stable per-file key: drop the timestamp, which differs between arms."""
    base = os.path.basename(path)
    return base.split("samples_")[1].rsplit("_20", 1)[0]


def load(d):
    """-> {task: {item_key: 0/1}}"""
    out = {}
    for f in glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True):
        task = task_of(f)
        bucket = out.setdefault(task, {})
        tag = subkey(f)
        for line in open(f):
            r = json.loads(line)
            s = r.get("acc_norm", r.get("acc"))
            v = 1.0 if float(s) >= 0.5 else 0.0
            key = r.get("doc_id")
            # mmlu subtasks reuse doc_id 0..n -> namespace by subtask name
            if task == "mmlu":
                key = (tag, key)
            bucket[key] = v
    return out


def main():
    dirs = sys.argv[1:]
    names = [os.path.basename(d.rstrip("/")).replace("paper_", "") for d in dirs]
    arms = [load(d) for d in dirs]
    tasks = sorted(set().union(*[set(a) for a in arms]),
                   key=lambda t: (t == "mmlu", t))
    print(f"arms: {', '.join(names)}   tasks: {len(tasks)}\n")

    header = f"{'task':<18}" + "".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header))
    per = {n: [] for n in names}
    pooled = {n: {} for n in names}
    for t in tasks:
        ids = None
        for a in arms:
            keys = set(a.get(t, {}))
            ids = keys if ids is None else (ids & keys)
        ids = sorted(ids)
        if not ids:
            continue
        row = f"{t:<18}"
        for n, a in zip(names, arms):
            k = sum(a[t][i] for i in ids)
            acc = k / len(ids)
            per[n].append(acc)
            for i in ids:
                pooled[n][(t, i)] = a[t][i]
            row += f"{acc:>10.4f}"
        print(row + f"   n={len(ids)}")

    print("-" * len(header))
    row = f"{'Avg8':<18}"
    avg = {}
    for n in names:
        avg[n] = sum(per[n]) / len(per[n])
        row += f"{avg[n]:>10.4f}"
    print(row + f"   ({len(per[names[0]])} tasks)")

    def mcnemar(a, b):
        common = set(pooled[a]) & set(pooled[b])
        x = sum(1 for k in common if pooled[a][k] == 1 and pooled[b][k] == 0)
        y = sum(1 for k in common if pooled[a][k] == 0 and pooled[b][k] == 1)
        p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
        return (x - y) / len(common), x, y, p

    print()
    ref = names[0]
    for n in names[1:]:
        d, x, y, p = mcnemar(n, ref)
        print(f"pooled {n:>6} - {ref:<6} d={d:+.4f}  discordant {x}/{y}  McNemar p={p:.2e}")
    if len(names) == 3:
        d, x, y, p = mcnemar(names[2], names[1])
        print(f"pooled {names[2]:>6} - {names[1]:<6} d={d:+.4f}  discordant {x}/{y}  "
              f"McNemar p={p:.2e}")


if __name__ == "__main__":
    main()
