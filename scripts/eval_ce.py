#!/usr/bin/env python
"""Held-out CE / token accuracy for one or more merged checkpoints.

Training loss is not comparable across runs (different effective steps, clip,
batch noise); this script scores every arm on the *same* token slices, one fixed
pass, no sampling. Use it for A/B decisions instead of `train_loss`.

  python scripts/eval_ce.py \
    --data data/fineweb_edu_holdout.jsonl \
    --models /path/LRC-1.5B-Base output/fw-dense-pt20M/v0-*/checkpoint-1220 \
             output/fw-corelr-k128-pt20M/v0-*/checkpoint-1220 \
    --seq-len 512 --max-docs 400 --device cuda --json-out output/eval_ce.json

Models are evaluated independently (batch 1, no padding), so tokenisation and
context length are identical for every arm.
"""
import argparse
import json
import math
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _label(path):
    """`.../v15-night/mod-gateup/0-5/v0-.../checkpoint-305` -> `mod-gateup/0-5/ckpt-305`."""
    parts = path.rstrip("/").split(os.sep)
    if len(parts) >= 4 and parts[-1].startswith("checkpoint-"):
        return f"{parts[-4]}/{parts[-3]}/{parts[-1].replace('checkpoint-', 'ckpt-')}"
    if len(parts) >= 3 and parts[-1].startswith("checkpoint-"):
        return f"{parts[-3]}/{parts[-1]}"
    return parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tokenizer", default=None, help="default: the first model")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-docs", type=int, default=400)
    ap.add_argument("--stride", type=int, default=0,
                    help=">0: sliding window stride over long docs (0 = first seq-len tokens only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    tok_dir = args.tokenizer or args.models[0]
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if len(texts) >= args.max_docs:
                break
    # tokenise once, share the exact same slices across models
    chunks = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) < 16:
            continue
        if args.stride > 0:
            for i in range(0, max(1, len(ids) - 8), args.stride):
                chunks.append(ids[i:i + args.seq_len])
        else:
            chunks.append(ids[:args.seq_len])
    print(f"[data] {len(texts)} docs -> {len(chunks)} chunks of <= {args.seq_len} tokens")

    rows = []
    for mid in args.models:
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        loss_sum = tok_correct = tok_total = 0
        per_chunk = []
        with torch.no_grad():
            for ids in chunks:
                x = torch.tensor([ids], device=dev)
                logits = model(input_ids=x, use_cache=False).logits.float()[:, :-1, :]
                y = x[:, 1:]
                n = int(y.numel())
                cl = float(F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
                loss_sum += cl
                per_chunk.append(cl / max(n, 1))
                tok_correct += int((logits.argmax(-1) == y).sum())
                tok_total += n
        ce = loss_sum / max(tok_total, 1)
        row = {"model": mid, "ce": ce, "ppl": math.exp(min(ce, 20.0)),
               "token_acc": tok_correct / max(tok_total, 1), "tokens": tok_total,
               "per_chunk_ce": per_chunk}
        rows.append(row)
        print(f"{_label(mid)}\n    CE={ce:.4f}  ppl={row['ppl']:.2f}  token_acc={row['token_acc']:.4f}  "
              f"tokens={tok_total}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if len(rows) > 1:
        print("\n[delta] paired over chunks; +-1.96*SE of the paired difference")
        base = rows[0]
        bp = base["per_chunk_ce"]
        for r in rows[1:]:
            d = [a - b for a, b in zip(r["per_chunk_ce"], bp)]
            n = len(d)
            m = sum(d) / max(n, 1)
            var = sum((x - m) ** 2 for x in d) / max(n - 1, 1)
            se = math.sqrt(var / n) if n > 1 else 0.0
            z = m / se if se > 0 else 0.0
            print(f"  {_label(r['model']):<26s} dCE={m:+.4f} +-{1.96 * se:.4f} (z={z:+.1f})  "
                  f"dAcc={r['token_acc'] - base['token_acc']:+.4f}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump({"config": vars(args), "rows": rows}, open(args.json_out, "w"), indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
