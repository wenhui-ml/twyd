"""Build a knowledge-dense PT corpus aimed at MMLU-style knowledge.

Why: docs/RESULTS_SUMMARY.md §4 lever 2. fineweb-edu is broad web text; our 524M-token
PT on it left MMLU at the base level (.4999 vs .5006). MMLU is textbook STEM /
humanities / social-science knowledge, so the PT corpus should look like that:

  * `wikimedia/wikipedia` (20231101.en) - encyclopedic coverage of every MMLU subject
  * `open-web-math/open-web-math`        - math-dense text (MMLU's STEM quarter)

Neither contains an MMLU/ARC/CSQA item (they are raw web/encyclopedia text), so this is
not eval contamination - it is domain matching.

    python scripts/build_pt_knowledge.py --target-gb 1.2 --out data/pt_knowledge.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def stream(dataset: str, name=None, split="train"):
    from datasets import load_dataset
    kw = dict(split=split, streaming=True)
    if name:
        kw["name"] = name
    return iter(load_dataset(dataset, **kw))


def text_of(row: dict) -> str:
    for k in ("text", "content", "raw_content"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pt_knowledge.jsonl")
    ap.add_argument("--target-gb", type=float, default=1.2)
    ap.add_argument("--wiki-frac", type=float, default=0.7)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    target = int(args.target_gb * 1e9)
    written = 0
    n = 0
    t0 = time.time()
    with open(args.out, "w") as fout:
        for tag, it, share in (
            ("wikipedia", stream("wikimedia/wikipedia", "20231101.en"), args.wiki_frac),
            ("open-web-math", stream("open-web-math/open-web-math"), 1.0 - args.wiki_frac),
        ):
            budget = int(target * share)
            got = 0
            print(f"[ptdata] {tag}: budget {budget/1e9:.2f} GB", flush=True)
            for row in it:
                t = text_of(row)
                if len(t) < args.min_chars:
                    continue
                if len(t) > args.max_chars:
                    t = t[: args.max_chars]
                line = json.dumps({"text": t}, ensure_ascii=False) + "\n"
                fout.write(line)
                got += len(line)
                written += len(line)
                n += 1
                if got >= budget:
                    break
                if n % 50000 == 0:
                    el = time.time() - t0
                    print(f"[ptdata] {n} docs  {written/1e9:.2f} GB  {written/1e6/max(el,1):.1f} MB/s",
                          flush=True)
            print(f"[ptdata] {tag}: done {got/1e9:.2f} GB", flush=True)
    print(f"[ptdata] TOTAL {n} docs  {written/1e9:.2f} GB (~{written/4e6:.0f}M tokens) -> {args.out}")


if __name__ == "__main__":
    main()
