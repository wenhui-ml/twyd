"""Generate teacher (Llama-3.2-3B-Instruct) CoT / MCQ data from our own corpus.

Purpose: the SFT arms in docs/EXPERIMENTS.md §4c-§4e trade behaviour gain against
MMLU damage and the net is a tie. The only SFT-stage lever left that can add
*knowledge* (not just format) is data written by the teacher: the student learns
facts from text it is asked to reproduce, which is the sequence-level distillation
route (Orca-style) rather than the PT-stage logit route.

Contamination: prompts come from `data/fw_score4_*.jsonl` (fineweb-edu), never from
the paper-8 eval sets, so no eval item is ever shown to the teacher. Generated items
are new text derived from public web educational passages.

    # throughput probe (32 prompts, one GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/gen_teacher_cot.py --limit 32 --benchmark

    # real shard, one process per GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/gen_teacher_cot.py --limit 20000 --out data/cot_shard0.jsonl
    CUDA_VISIBLE_DEVICES=1 python scripts/gen_teacher_cot.py --offset 20000 --limit 20000 --out data/cot_shard1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 60% multiple-choice-with-reasoning (closest to the eval format), 40% study notes.
MCQ_PROMPT = (
    "Do two things with the passage below.\n"
    "1. Write ONE exam-style multiple-choice question that tests a specific fact it states.\n"
    "   Put exactly four options on their own lines, labelled (A) (B) (C) (D).\n"
    "2. On the next line write 'Answer: <letter>', then ONE sentence of reasoning.\n"
    "Use only facts from the passage. Do not invent extra questions.\n\n"
    "Passage:\n{passage}\n\nOutput:"
)
NOTE_PROMPT = (
    "Read the passage and write a compact study note a student could revise from.\n"
    "Rules:\n- 4-6 bullet points, each a self-contained fact in your own words.\n"
    "- One sentence on why it matters and one common misconception it corrects.\n\n"
    "Passage:\n{passage}\n\nStudy note:"
)


def build_prompts(limit: int, offset: int, passage_chars: int, seed: int = 42):
    path = os.environ.get("COT_CORPUS", "data/fw_score4_1B.jsonl")
    rng = random.Random(seed)
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(out) >= limit:
                break
            try:
                text = json.loads(line).get("text") or ""
            except Exception:
                continue
            text = text.strip()
            if len(text) < 400:
                continue
            out.append(text[:passage_chars])
    rng.shuffle(out)
    prompts = []
    for i, passage in enumerate(out):
        if i % 5 < 3:
            prompts.append(MCQ_PROMPT.format(passage=passage))
        else:
            prompts.append(NOTE_PROMPT.format(passage=passage))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/chenwh/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--passage-chars", type=int, default=2400)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--out", default="")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=os.environ.get("COT_ATTN", "sdpa"))
    model.eval()

    prompts = build_prompts(args.limit, args.offset, args.passage_chars)
    print(f"[gen] {len(prompts)} prompts, batch={args.batch}, new<={args.max_new_tokens}", flush=True)

    fout = open(args.out, "w") if args.out else None
    t0 = time.time()
    n_out_tokens = 0
    n_done = 0
    for start in range(0, len(prompts), args.batch):
        chunk = prompts[start:start + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1400).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.08,
                                 num_beams=1, pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        n_out_tokens += int((new_tokens != tok.pad_token_id).sum())
        for prompt, answer in zip(chunk, texts):
            answer = answer.strip()
            if not answer:
                continue
            n_done += 1
            if fout is not None:
                user = prompt.split("Passage:")[0].strip()
                fout.write(json.dumps({"messages": [
                    {"role": "user", "content": user + "\n\nPassage:\n" +
                     prompt.split("Passage:", 1)[1].rsplit("\n\n", 1)[0].strip()},
                    {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
        if start % (args.batch * 10) == 0:
            el = time.time() - t0
            print(f"[gen] {n_done}/{len(prompts)}  {n_out_tokens/el:.0f} tok/s  "
                  f"{el:.0f}s", flush=True)
    el = time.time() - t0
    if fout is not None:
        fout.close()
    print(f"[gen] DONE {n_done} samples, {n_out_tokens} tokens in {el:.0f}s "
          f"= {n_out_tokens/max(el,1e-9):.0f} tok/s", flush=True)
    if args.benchmark:
        print(f"[gen] ETA for 100M output tokens on this GPU: "
              f"{100e6/max(n_out_tokens/max(el,1e-9),1e-9)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
