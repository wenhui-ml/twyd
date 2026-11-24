#!/usr/bin/env python3
"""Stream ungated Cosmopedia textbook corpora into a local knowledge JSONL pool.

Combines:
  - HuggingFaceTB/cosmopedia (openstax):      college textbooks (STEM / Econ / Psych / Law / Bio)
  - HuggingFaceTB/cosmopedia (stanford):      university lecture courses
  - HuggingFaceTB/cosmopedia (auto_math_text): math reasoning & problem solving
  - Existing data/pt_knowledge.jsonl (wiki + open-web-math)
before feeding into scripts/build_manifold_pt_corpus.py for neighbor-window packing.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pt_textbook_pool.jsonl")
    ap.add_argument("--extra-knowledge", default="data/pt_knowledge.jsonl")
    ap.add_argument("--openstax-mb", type=float, default=220.0)
    ap.add_argument("--stanford-mb", type=float, default=130.0)
    ap.add_argument("--math-mb", type=float, default=100.0)
    ap.add_argument("--khan-mb", type=float, default=0.0)
    ap.add_argument("--webv2-mb", type=float, default=0.0)
    ap.add_argument("--skip-openstax-mb", type=float, default=0.0)
    ap.add_argument("--skip-stanford-mb", type=float, default=0.0)
    ap.add_argument("--skip-math-mb", type=float, default=0.0)
    ap.add_argument("--extra-mb", type=float, default=150.0)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    from datasets import load_dataset

    t0 = time.time()
    n_total = 0
    b_total = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for cfg, mb, skip_mb in (
            ("openstax", args.openstax_mb, args.skip_openstax_mb),
            ("stanford", args.stanford_mb, args.skip_stanford_mb),
            ("khanacademy", args.khan_mb, 0.0),
            ("auto_math_text", args.math_mb, args.skip_math_mb),
            ("web_samples_v2", args.webv2_mb, 0.0),
        ):
            if mb <= 0:
                continue
            budget = int(mb * 1e6)
            skip_budget = int(skip_mb * 1e6)
            skipped = 0
            got = 0
            cnt = 0
            print(f"[textbook] streaming cosmopedia/{cfg} skip={skip_mb:.0f} MB target={mb:.0f} MB ...", flush=True)
            ds = load_dataset("HuggingFaceTB/cosmopedia", name=cfg, split="train", streaming=True)
            for row in ds:
                txt = (row.get("text") or "").strip()
                if len(txt) < args.min_chars:
                    continue
                if len(txt) > args.max_chars:
                    txt = txt[: args.max_chars]
                line = json.dumps({"text": txt}, ensure_ascii=False) + "\n"
                lb = len(line.encode("utf-8"))
                if skipped < skip_budget:
                    skipped += lb
                    continue
                fout.write(line)
                got += lb
                b_total += lb
                cnt += 1
                n_total += 1
                if cnt % 10000 == 0:
                    print(f"[textbook] {cfg}: {cnt} docs  {got/1e6:.1f}/{mb:.0f} MB  ({time.time()-t0:.1f}s)",
                          flush=True)
                if got >= budget:
                    break
            print(f"[textbook] {cfg} done: {cnt} docs  {got/1e6:.1f} MB (skipped {skipped/1e6:.1f} MB)", flush=True)

        if args.extra_knowledge and os.path.isfile(args.extra_knowledge) and args.extra_mb > 0:
            ebudget = int(args.extra_mb * 1e6)
            egot = 0
            ecnt = 0
            with open(args.extra_knowledge, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    lb = len(line.encode("utf-8"))
                    egot += lb
                    b_total += lb
                    ecnt += 1
                    n_total += 1
                    if egot >= ebudget:
                        break
            print(f"[textbook] extra {args.extra_knowledge}: {ecnt} docs  {egot/1e6:.1f} MB", flush=True)

    print(f"[textbook] TOTAL pool: {n_total} docs  {b_total/1e6:.1f} MB -> {args.out} in {time.time()-t0:.1f}s",
          flush=True)
    # Streaming HF datasets can hang at interpreter teardown waiting on HTTP/pyarrow threads.
    os._exit(0)


if __name__ == "__main__":
    main()
