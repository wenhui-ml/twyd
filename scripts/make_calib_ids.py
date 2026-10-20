#!/usr/bin/env python
"""Pack a text/JSONL corpus into calibration token ids for calib_ridge.py.

Usage:
  python scripts/make_calib_ids.py \
    --tokenizer /path/to/teacher --input calib.jsonl \
    --num-seqs 200 --seq-len 1024 --out outputs/calib_ids.pt
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoTokenizer

from twyd.init import pack_calib_ids


def _texts(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        items = [json.loads(line) for line in open(path) if line.strip()]
    elif ext == ".json":
        obj = json.load(open(path))
        items = obj if isinstance(obj, list) else [obj]
    else:
        return [line.strip() for line in open(path) if line.strip()]
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append(it.get("text") or it.get("content") or it.get("prompt") or "")
        else:
            out.append(str(it))
    return [t for t in out if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True, help=".jsonl/.json/.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    texts = _texts(args.input)
    if not texts:
        raise SystemExit(f"no text found in {args.input}")
    seqs = [tok(t, add_special_tokens=False).input_ids for t in texts]
    ids = pack_calib_ids(seqs, seq_len=args.seq_len, num_seqs=args.num_seqs,
                         eos_id=getattr(tok, "eos_token_id", None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"input_ids": ids}, args.out)
    print(f"[calib] wrote {tuple(ids.shape)} ids to {args.out}")


if __name__ == "__main__":
    main()
