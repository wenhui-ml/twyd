#!/usr/bin/env python
"""Build a PT corpus with a fixed share of instruction data.

Why: SFT cannot recover the MMLU the PT stage costs (100M chat / 200M chat / 200M
50-50 mix all land at MMLU 0.486 against base 0.501), so the next lever is the PT
*distribution*. This writes {"text": ...} lines = fineweb-edu interleaved with
rendered instruction/chat documents at a fixed byte ratio, so the share holds no
matter how the loader orders the stream (shuffled or not).

  python scripts/build_pt_mix.py \
      --fineweb data/fw_score4_1B.jsonl \
      --instr    data/sft_mix_200M.jsonl \
      --out      data/fw_score4_1B_mix10.jsonl \
      --instr-ratio 0.10 \
      --tokenizer /home/chenwh/models/LRC-1.5B-Base
"""
import argparse
import json
import os
import re
import sys

# The student's tokenizer in a merged LRC checkpoint carries the placeholder
# "Entry not found" as its chat_template, so render with the teacher tokenizer and
# strip the system block it injects (a today's-date banner is noise for PT).
_SYS = re.compile(r"^\s*(<\|begin_of_text\|>)?\s*<\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>",
                  re.S)


def render(messages, tok, max_chars):
    """messages -> one training string. Chat template when available."""
    msgs = [m for m in messages if m.get("content")]
    if not msgs:
        return None
    plain = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m['content']}" for m in msgs
    )
    content_len = sum(len(m["content"]) for m in msgs)
    text, why = None, "template"
    if tok is not None:
        try:
            cand = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            cand = _SYS.sub(r"\1", cand, count=1)
            # a broken template returns a stub ("Entry not found") -- reject it
            if len(cand) >= 0.5 * content_len:
                text = cand
            else:
                why = f"template too short ({len(cand)} vs {content_len})"
        except Exception as e:
            why = f"template failed ({e})"
    if text is None:
        text = plain
        why += " -> plain fallback"
    return (text[:max_chars] if max_chars and len(text) > max_chars else text), why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instr-ratio", type=float, default=0.10,
                    help="target share of rendered instruction TEXT by bytes")
    ap.add_argument("--tokenizer", default=None, help="dir with tokenizer/chat template")
    ap.add_argument("--max-instr-chars", type=int, default=16000,
                    help="cap one instruction doc (long agent traces otherwise dominate)")
    ap.add_argument("--limit-instr-docs", type=int, default=0)
    ap.add_argument("--limit-fineweb", type=int, default=0,
                    help="smoke test: stop after N fineweb docs")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            print(f"[mix] tokenizer {args.tokenizer} chat_template={bool(tok.chat_template)}")
        except Exception as e:
            print(f"[mix] tokenizer unavailable ({e}); falling back to plain rendering")

    # --- pass 1: collect rendered instruction docs until the byte target is met.
    fw_text_bytes = os.path.getsize(args.fineweb)
    target = int(fw_text_bytes * args.instr_ratio / (1.0 - args.instr_ratio))
    print(f"[mix] fineweb file {fw_text_bytes/1e9:.2f} GB -> instruction target "
          f"{target/1e6:.0f} MB (ratio {args.instr_ratio:.2f})")

    instr_texts, got, n_bad = [], 0, 0
    with open(args.instr, "r", encoding="utf-8") as f:
        for line in f:
            if got >= target:
                break
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            t, why = render(obj.get("messages") or [], tok, args.max_instr_chars)
            if not t:
                n_bad += 1
                continue
            if len(instr_texts) == 0:
                print(f"[mix] first doc rendered by {why}: {len(t)} chars")
            instr_texts.append(t)
            got += len(t.encode("utf-8"))
            if args.limit_instr_docs and len(instr_texts) >= args.limit_instr_docs:
                break
    if not instr_texts:
        sys.exit("!! no instruction documents rendered")
    print(f"[mix] {len(instr_texts)} instruction docs, {got/1e6:.0f} MB rendered "
          f"({n_bad} skipped), mean {got/max(1,len(instr_texts)):.0f} B/doc")

    # --- pass 2: interleave by bytes so the ratio is uniform along the stream.
    step = (got / len(instr_texts)) / args.instr_ratio * (1.0 - args.instr_ratio)  # fineweb B per instr doc
    size = (got / len(instr_texts)) / args.instr_ratio                             # fineweb B + instr B per block
    print(f"[mix] interleave: one instruction doc per {step/1e6:.3f} MB of fineweb "
          f"(block {size/1e6:.3f} MB)")

    written_fw = written_in = 0
    acc, ii = 0.0, 0
    n_out = 0
    with open(args.fineweb, "r", encoding="utf-8") as src, \
            open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            n_out += 1
            written_fw += 1
            if args.limit_fineweb and written_fw >= args.limit_fineweb:
                break
            acc += len(line.encode("utf-8"))
            if ii < len(instr_texts) and acc >= step:
                acc -= step
                dst.write(json.dumps({"text": instr_texts[ii]}, ensure_ascii=False) + "\n")
                n_out += 1
                written_in += 1
                ii += 1
            if written_in >= len(instr_texts):
                # ratio satisfied for the whole stream; keep the rest of fineweb only
                pass
    ratio = written_in / max(1, n_out)
    print(f"[mix] wrote {args.out}: {n_out} docs ({written_fw} fw + {written_in} instr), "
          f"doc ratio {ratio:.3f}")
    print(f"[mix] done {os.path.getsize(args.out)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
