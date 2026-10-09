"""Standalone Dense-LRC distillation trainer (no ms-swift dependency).

Teacher: Llama-3.2-1B (H=2048, d_ff=8192, rho=4.0). Student: dense continuation
of a sliced/inherited init at width r (default 1024).
Loss (paper A.5 Eq.6): T^2*KL(T=40) + CE, aux alignment optional/stripped (GPD).

Usage (swift env):
  python scripts/train_dense_standalone.py --teacher /home/chenwh/models/Llama-3.2-1B \\
      --output /home/chenwh/Papers/twyd/outputs/dense-1b --steps 500
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_student_from_teacher(teacher, r: int):
    import copy
    cfg = copy.deepcopy(teacher.config)
    cfg.hidden_size = r
    cfg.intermediate_size = teacher.config.intermediate_size  # inherit d_ff (LRC style)
    cfg.num_attention_heads = max(4, teacher.config.num_attention_heads // 2)
    cfg.num_key_value_heads = max(2, teacher.config.num_key_value_heads // 2)
    student = AutoModelForCausalLM.from_config(cfg)
    # inherit sliced weights where shapes allow (dense warm start at merged LRC point)
    with torch.no_grad():
        tsd = dict(teacher.named_parameters())
        for n, p in student.named_parameters():
            if n in tsd and tsd[n].shape == p.shape:
                p.copy_(tsd[n])
            elif n in tsd:
                t = tsd[n]
                # slice last-two dims to student shape
                idx = tuple(slice(0, s) for s in p.shape)
                try:
                    p.copy_(t[idx])
                except Exception:
                    pass
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--student-hidden", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    ms_id = args.teacher
    local_teacher = args.teacher if os.path.isdir(args.teacher) else None
    if local_teacher is None:
        try:
            from modelscope import snapshot_download
            local_teacher = snapshot_download(ms_id)
        except Exception:
            local_teacher = ms_id
    tok = AutoTokenizer.from_pretrained(local_teacher, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        local_teacher, dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_student_from_teacher(teacher, args.student_hidden).to(torch.bfloat16).to(args.device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    vocab = teacher.config.vocab_size
    log = []
    for step in range(1, args.steps + 1):
        ids = torch.randint(0, vocab, (args.batch, args.seq_len), device=args.device)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=attn)
        s_out = student(input_ids=ids, attention_mask=attn)
        s_shift = s_out.logits[..., :-1, :].contiguous()
        t_shift = t_out.logits[..., :-1, :].contiguous().detach()
        y_shift = ids[..., 1:].contiguous()
        T = args.temperature
        kl = F.kl_div(F.log_softmax(s_shift / T, -1), F.softmax(t_shift / T, -1),
                      reduction="batchmean") * (T * T)
        ce = F.cross_entropy(s_shift.view(-1, s_shift.size(-1)), y_shift.view(-1))
        loss = (kl + ce) / args.grad_accum
        loss.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if step % 25 == 0:
            rec = {"step": step, "kl": float(kl.detach()), "ce": float(ce.detach())}
            log.append(rec)
            print(rec, flush=True)
    student.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    with open(os.path.join(args.output, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "log": log}, f, indent=2)
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
