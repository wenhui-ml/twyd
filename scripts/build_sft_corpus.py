"""Convert ultrachat_200k into the chat jsonl ms-swift expects.

Accepts either layout:
  * parquet with a `messages` column (HF style)
  * csv with `prompt_id,prompt,messages` where `messages` is a JSON string (ModelScope mirror)

Output lines: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
swift's template masks the prompt, so only assistant turns are supervised.

  python scripts/build_sft_corpus.py --src /run/media/.../sft/data \
      --out data/sft_ultrachat_200k.jsonl --max-tokens 1e8
"""
import argparse
import ast
import glob
import json

import pyarrow.csv as pcsv
import pyarrow.parquet as pq


def _parse_messages(raw):
    """The ModelScope CSV stores the list as a *python repr* (single quotes), not JSON."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


def conversations(path):
    """Yield message lists from one file, whatever the container."""
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=["messages"]):
            yield from batch.to_pydict()["messages"]
        return
    # pandas' C engine handles the embedded newlines/quotes that pyarrow.csv rejects.
    import pandas as pd
    for chunk in pd.read_csv(path, usecols=["messages"], chunksize=2000,
                             engine="c", on_bad_lines="skip"):
        for raw in chunk["messages"]:
            parsed = _parse_messages(raw)
            if parsed:
                yield parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the ultrachat files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=float, default=1e8, help="0 = no cap")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="keep at most this many messages per conversation (0 = all)")
    ap.add_argument("--include", default="train_sft", help="filename substring filter")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(f"{args.src}/**/*.*", recursive=True)
                   if args.include in f and (f.endswith(".parquet") or f.endswith(".csv")))
    if not files:
        raise SystemExit(f"no parquet/csv with '{args.include}' under {args.src}")

    cap = int(args.max_tokens)
    rows = chars = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        stop = False
        for path in files:
            for msgs in conversations(path):
                if not msgs:
                    continue
                conv = [{"role": m.get("role", "user"),
                         "content": (m.get("content") or "").strip()}
                        for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                if len(conv) < 2:
                    continue
                if args.max_turns:
                    conv = conv[:args.max_turns]
                fout.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
                rows += 1
                chars += sum(len(m["content"]) for m in conv)
                if cap and chars / 4 >= cap:      # ~4 chars per token
                    stop = True
                    break
            if stop:
                break
    print(f"[sft] {args.out}: conversations={rows:,} approx_tokens={chars/4/1e6:.1f}M "
          f"from {len(files)} file(s)")


if __name__ == "__main__":
    main()
