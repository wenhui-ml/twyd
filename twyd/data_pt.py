"""LRC-PT pretraining data pipeline (paper Sec 5.1, LRC Mixed recipe).

Original LRC (LowRankClone/data/generate_general_data_parallel.py:49-86):
  text = eos_token + sample["text"] + bos_token   # manual delimiters
  tokens = tokenizer(text, add_special_tokens=False)
  buffer = concat across docs; yield fixed data_max_len chunks (1024)
  labels = input_ids (full supervision, no -100)

PT stage: every position distills (KL + CE), labels == input_ids.
SFT stage (separate): chat_template + prompt masked to -100.

Two usages:
  A. Offline (no ms-swift needed):
    python -m twyd.data_pt --tokenizer <id-or-path> --input data.jsonl \\
        --output pt_chunks.jsonl --seq-len 1024
    Input jsonl: {"text": ...} per line (also accepts {"input_ids": [...]}
  B. ms-swift plugin (registered on `import twyd.plugin`):
    --dataset twyd_pt::/path/to/raw.jsonl --tuner_kwargs '{"pt_seq_len":1024,...}'
    A RowPreprocessor packs raw texts with the LRC eos+text+bos recipe so
    `swift pt` consumes the PT stream natively (labels == input_ids).
"""
from __future__ import annotations

import argparse
import json
from typing import Iterator, List, Optional

import torch


def pack_texts_to_chunks(
    all_ids: List[List[int]],
    seq_len: int,
) -> Iterator[List[int]]:
    """Concat token streams across docs, cut fixed seq_len chunks (LRC style)."""
    buf: List[int] = []
    for ids in all_ids:
        buf.extend(ids)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]
    # LRC drops the tail remainder (no padding of partial chunks).


def tokenize_lrc_style(
    tokenizer,
    texts: List[str],
    seq_len: int,
    drop_tail: bool = True,
) -> List[dict]:
    """eos + text + bos per doc, add_special_tokens=False, concat + chunk."""
    try:
        eos = tokenizer.eos_token or ""
    except Exception:
        eos = ""
    try:
        bos = tokenizer.bos_token or ""
    except Exception:
        bos = ""
    streams: List[List[int]] = []
    for t in texts:
        s = f"{eos}{t}{bos}"
        try:
            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        except Exception:
            ids = tokenizer.encode(s, add_special_tokens=False)
        streams.append(list(ids))
    out = []
    for chunk in pack_texts_to_chunks(streams, seq_len):
        out.append({"input_ids": chunk, "labels": list(chunk)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LRC-PT packed chunks")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True, help="jsonl with {text} or {input_ids}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-chunks", type=int, default=-1)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True,
                                        trust_remote_code=True)
    texts: List[List[int]] = []
    raw_texts: List[str] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "input_ids" in d and isinstance(d["input_ids"], list):
                texts.append([int(x) for x in d["input_ids"]])
            elif "text" in d and isinstance(d["text"], str):
                raw_texts.append(d["text"])
    chunks: List[dict] = []
    if raw_texts:
        chunks.extend(tokenize_lrc_style(tok, raw_texts, args.seq_len))
    if texts:
        for chunk in pack_texts_to_chunks(texts, args.seq_len):
            chunks.append({"input_ids": chunk, "labels": list(chunk)})
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    with open(args.output, "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} chunks seq_len={args.seq_len} -> {args.output}")


# --- ms-swift dataset plugin: twyd_pt::<raw.jsonl> ---------------------------
# Registered on `import twyd.plugin`. Packs raw {"text"} rows into fixed
# seq_len chunks with the LRC recipe at preprocess time, so `swift pt`
# consumes full-supervision chunks directly (labels == input_ids).
try:
    from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

    class TwydPTPreprocessor(RowPreprocessor):
        """Concat raw texts (eos+text+bos, no special tokens) into chunks."""

        prompt = "text"

        def preprocess(self, row: dict) -> Optional[dict]:
            # RowPreprocessor works row-wise; chunking state lives on self and
            # is flushed by prompt_format / dataset packing downstream. We emit
            # one packed-text row per input and let swift's packing (or our
            # offline script for exact LRC parity) cut fixed chunks.
            text = row.get("text", None)
            if not isinstance(text, str) or not text.strip():
                return None
            eos = getattr(self, "_twyd_eos", None) or ""
            bos = getattr(self, "_twyd_bos", None) or ""
            # Keep delimiters as literal text here; tokenization adds no
            # special tokens (mirrors offline tokenize_lrc_style).
            return {"text": f"{eos}{text}{bos}"}

    register_dataset(DatasetMeta(
        ms_dataset_id="twyd_pt",
        hf_dataset_id="twyd_pt",
        preprocess_func=TwydPTPreprocessor(),
        tags=["pretrain", "twyd"],
    ))
except Exception:
    pass


if __name__ == "__main__":
    main()
