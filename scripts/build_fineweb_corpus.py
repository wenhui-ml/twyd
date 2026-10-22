#!/usr/bin/env python
"""Build a FineWeb-Edu pre-training corpus (jsonl of {"text": ...}) from local parquet.

The parquet files carry `int_score` (0-5 FineWeb-Edu quality) and `token_count`, so
filtering by quality and budgeting by tokens is exact — no tokenizer pass needed.

  # first 60M tokens at score>=4 (what the current 20M-token arms train on)
  python scripts/build_fineweb_corpus.py --max-tokens 60e6 --out data/fw_score4_60M.jsonl

  # extend it: skip the docs already used, take the next 69.5M tokens
  python scripts/build_fineweb_corpus.py --skip-docs 51390 --max-tokens 69.5e6 \
      --out data/fw_score4_next.jsonl

`--skip-docs` counts *qualifying* documents, so a holdout built with the same
flag set is guaranteed disjoint from the training corpus.
"""
import argparse
import glob
import json

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/fineweb_edu/**/*.parquet",
                    help="parquet glob (recursive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = no doc cap")
    ap.add_argument("--max-tokens", type=float, default=0.0, help="0 = no token cap")
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="skip this many qualifying docs first (for holds-outs / appends)")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all shards")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"no parquet matched {args.glob}")

    max_tokens = int(args.max_tokens)
    seen = kept = rows = 0
    tokens = 0
    with open(args.out, "w", encoding="utf-8") as f:
        stop = False
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2000, columns=["text", "int_score", "token_count"]):
                d = batch.to_pydict()
                for text, sc, tc in zip(d["text"], d["int_score"], d["token_count"]):
                    if sc is None or sc < args.min_score or not text:
                        continue
                    seen += 1
                    if kept < args.skip_docs:
                        kept += 1
                        continue
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    rows += 1
                    tokens += int(tc or 0)
                    if (args.max_docs and rows >= args.max_docs) or \
                       (max_tokens and tokens >= max_tokens):
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
    print(f"[corpus] {args.out}: docs={rows:,} tokens={tokens/1e6:.1f}M "
          f"(scanned {seen:,} qualifying docs, skipped {args.skip_docs:,}) "
          f"from {len(files)} shard(s)")


if __name__ == "__main__":
    main()
