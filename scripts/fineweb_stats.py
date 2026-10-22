#!/usr/bin/env python
"""Inventory local FineWeb-Edu parquet shards: rows, tokens, and score>=4 share.

Reads only `int_score` and `token_count`, so it is ~50x cheaper than a text scan.

  python scripts/fineweb_stats.py --glob '/run/media/chenwh/.../fineweb-edu/**/*.parquet'
"""
import argparse
import glob

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/fineweb_edu/**/*.parquet")
    ap.add_argument("--min-score", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise SystemExit(f"no parquet matched {args.glob}")
    tot_rows = tot_tok = tot_n = tot_ntok = 0
    print(f"{'shard':<52s} {'rows':>10s} {'tokens':>9s} {'score>=' + str(args.min_score):>22s}")
    for p in files:
        pf = pq.ParquetFile(p)
        rows = tok = n = ntok = 0
        for b in pf.iter_batches(batch_size=100_000, columns=["int_score", "token_count"]):
            d = b.to_pydict()
            rows += len(d["int_score"])
            tok += sum(x or 0 for x in d["token_count"])
            for sc, tc in zip(d["int_score"], d["token_count"]):
                if sc is not None and sc >= args.min_score:
                    n += 1
                    ntok += int(tc or 0)
        tot_rows += rows
        tot_tok += tok
        tot_n += n
        tot_ntok += ntok
        short = "/".join(p.split("/")[-2:])
        print(f"{short:<52s} {rows:>10,} {tok / 1e6:>8.1f}M "
              f"{n:>10,} ({100 * n / max(rows, 1):5.1f}%) {ntok / 1e6:>6.1f}M")
    print(f"{'TOTAL (' + str(len(files)) + ' shards)':<52s} {tot_rows:>10,} {tot_tok / 1e6:>8.1f}M "
          f"{tot_n:>10,} ({100 * tot_n / max(tot_rows, 1):5.1f}%) {tot_ntok / 1e6:>6.1f}M")


if __name__ == "__main__":
    main()
