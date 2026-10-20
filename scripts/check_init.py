#!/usr/bin/env python
"""Step-0 gate for a TWYD student: is the initialized width-r model alive?

The 2026-10 run (`dense-lora128-r1024-ddp-2gpu-100k`) started from a
top-left slice of Llama-3.2-3B and was already random at step 1
(``loss=152``, ``token_acc=1.9e-4``); 1368 steps then only reached 13% token
accuracy and every eval (HellaSwag 25.8, ARC-e 25.6, PIQA 51.9, MMLU 26.2) was
at chance. This script makes that failure visible before spending GPU hours.

It builds exactly what ``_prepare_model_impl`` builds (shrink + factorized
student + freeze) and reports:

  * CE / token accuracy of teacher and student on the same token slice
  * per-layer cosine between student hidden states and the teacher's first r
    dimensions (the slice init should track the teacher; it collapses by layer
    4 in the broken run)
  * attention/GQA layout and trainable parameter breakdown

Usage:
  python scripts/check_init.py \
      --teacher /home/chenwh/models/Llama-3.2-3B-Instruct \
      --model   /home/chenwh/models/Llama-3.2-3B-Instruct \
      --r 1536 --branch dense --seq-len 256

Exit code is 1 when the step-0 gate fails (CE > --max-ce or mean cos < --min-cos).
"""
import argparse
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXT = (
    "The capital of France is Paris. Water boils at 100 degrees Celsius at sea level. "
    "In mathematics, the derivative of x squared is 2x. The quick brown fox jumps over "
    "the lazy dog. Large language models are trained on vast amounts of text data to "
    "predict the next token. A well-initialized compressed student should already track "
    "its teacher on ordinary text before any distillation step is taken."
)


def _cfg(mod):
    return getattr(getattr(mod, "config", None), "text_config", None) or getattr(mod, "config", None)


def _ce_acc(logits, ids):
    logits = logits.float()[:, :-1, :]
    y = ids[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return ce, acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--model", default=None, help="student base passed to swift --model (default: teacher)")
    ap.add_argument("--r", type=int, default=1536, help="student hidden width (paper: 1536 for 3B)")
    ap.add_argument("--branch", default="dense", choices=["dense", "core", "plain"])
    ap.add_argument("--lora-rank", type=int, default=0, help="0 = full dense (paper); >0 = LoRA variant")
    ap.add_argument("--warm-start", default="slice", choices=["slice", "ridge"])
    ap.add_argument("--master-p", default="none", choices=["none", "embed_svd"],
                    help="project embedding/attention/MLP onto embedding-SVD P")
    ap.add_argument("--head-mode", default="select", choices=["select", "merge"],
                    help="P attention head mapping: select first heads or average groups")
    ap.add_argument("--ridge-perp", default="residual", choices=["zero", "residual", "lowrank"])
    ap.add_argument("--calib-ids", default=None, help=".pt/.npy/.jsonl/.txt for --warm-start ridge")
    ap.add_argument("--calib-seq-len", type=int, default=256)
    ap.add_argument("--calib-num-seqs", type=int, default=32)
    ap.add_argument("--calib-attn", action="store_true",
                    help="also ridge q/k/v/o with head-projected teacher targets")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ce", type=float, default=6.0)
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    from twyd.config import TwydLRCConfig
    from twyd.init import (build_master_P, calibrate_attention,
                           project_student_with_P_)
    from twyd.tuner import (_TwydWrapper, _build_factorized_student,
                            _freeze_for_branch, _hidden_of, _shrink_student_,
                            _student_head_layout)

    dev = torch.device(args.device)
    model_id = args.model or args.teacher
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    student = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    H = _hidden_of(teacher)
    r = int(args.r)
    tcfg = _cfg(teacher)
    student_hidden = _hidden_of(student)
    if student_hidden != r:
        heads, kv, head_dim, new_heads, new_kv, kv_map = _student_head_layout(tcfg, r)
        print(f"[layout] teacher hidden={H} heads={heads} kv={kv} head_dim={head_dim}")
        print(f"[layout] width shrink -> r={r} heads={new_heads} kv={new_kv} kv_map={kv_map}")
        student = _shrink_student_(student, teacher, r)
    else:
        scfg = _cfg(student)
        print(f"[layout] student loaded as-is: hidden={student_hidden} "
              f"heads={getattr(scfg, 'num_attention_heads', '?')} "
              f"kv={getattr(scfg, 'num_key_value_heads', '?')} "
              f"head_dim={getattr(scfg, 'head_dim', '?')} "
              f"d_ff={getattr(scfg, 'intermediate_size', '?')}")
    P = None
    if args.master_p != "none":
        P = build_master_P(teacher, r)
        n_p = project_student_with_P_(student, teacher, P, head_mode=args.head_mode)
        print(f"[master-p] P={tuple(P.shape)} projected {n_p} tensors")
    cfg = TwydLRCConfig(branch=args.branch, teacher_model=args.teacher, student_hidden=r,
                        stage="pt", warm_start=args.warm_start,
                        ridge_perp=args.ridge_perp, calib_ids=args.calib_ids,
                        calib_seq_len=args.calib_seq_len,
                        calib_num_seqs=args.calib_num_seqs,
                        lora_rank=args.lora_rank,
                        target_modules=["gate_proj", "up_proj", "down_proj"])
    cfg.validate()
    meta = _build_factorized_student(student, teacher, cfg, branch=args.branch)
    wrapper = _TwydWrapper(student, teacher, cfg)
    wrapper.twyd_cfg = cfg
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)
    if args.warm_start == "ridge":
        if not args.calib_ids:
            print("[ridge] --warm-start ridge needs --calib-ids; keeping slice init")
        else:
            from twyd.init import calibrate_ridge, load_calib_ids
            ids = load_calib_ids(args.calib_ids, tokenizer=tok,
                                 seq_len=args.calib_seq_len,
                                 num_seqs=args.calib_num_seqs)
            if args.calib_attn:
                P_attn = P if P is not None else torch.eye(H)[:r]
                rep_a = calibrate_attention(
                    student, teacher, ids, P=P_attn, head_mode=args.head_mode,
                    ridge_lambda=cfg.ridge_lambda, batch_size=2)
                print(f"[attn] groups={rep_a['n_groups']} "
                      f"rel {rep_a['mean_rel_before']:.4f} -> "
                      f"{rep_a['mean_rel_deployed']:.4f}")
            rep = calibrate_ridge(student, teacher, ids,
                                  target_modules=cfg.target_modules,
                                  ridge_lambda=cfg.ridge_lambda,
                                  perp=cfg.ridge_perp, joint=True,
                                  batch_size=2)
            print(f"[ridge] groups={rep['n_groups']} tokens={rep['n_tokens']} "
                  f"rel {rep['mean_rel_before']:.4f} -> "
                  f"{rep.get('mean_rel_deployed', rep['mean_rel_after']):.4f}")

    trainable = [(n, tuple(p.shape)) for n, p in student.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[params] trainable tensors={len(trainable)} params={n_train/1e6:.1f}M")
    for n, s in trainable[:8]:
        print("   ", n, s)

    text = args.text or DEFAULT_TEXT
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :args.seq_len]
    print(f"[data] {ids.shape[1]} tokens")

    teacher = teacher.to(dev)
    student = student.to(dev)
    with torch.no_grad():
        t_out = teacher(input_ids=ids.to(dev), use_cache=False, output_hidden_states=True)
        t_ce, t_acc = _ce_acc(t_out.logits, ids.to(dev))
        ths = [h.float() for h in t_out.hidden_states]

    student.eval()
    with torch.no_grad():
        s_out = student(input_ids=ids.to(dev), use_cache=False, output_hidden_states=True)
        s_ce, s_acc = _ce_acc(s_out.logits, ids.to(dev))
        shs = [h.float() for h in s_out.hidden_states]

    print(f"[teacher] CE={t_ce:.3f} tok_acc={t_acc:.4f}")
    print(f"[student] CE={s_ce:.3f} tok_acc={s_acc:.4f}")

    cos_all = []
    step = max(1, len(shs) // 8)
    for l in range(0, min(len(shs), len(ths)), step):
        s = shs[l]
        if P is not None:
            t = ths[l] @ P.T.to(device=ths[l].device, dtype=ths[l].dtype)
        else:
            t = ths[l][..., :s.shape[-1]]
        cos = F.cosine_similarity(s.reshape(-1, s.shape[-1]),
                                  t.reshape(-1, t.shape[-1]), dim=-1).mean().item()
        rel = ((s - t).norm() / t.norm()).item()
        cos_all.append(cos)
        print(f"[layer {l:2d}] cos={cos:.4f} relerr={rel:.4f} "
              f"|s|={s.norm(dim=-1).mean():.2f} |t|={t.norm(dim=-1).mean():.2f}")
    mean_cos = sum(cos_all) / max(len(cos_all), 1)

    ok = (s_ce <= args.max_ce) and (mean_cos >= args.min_cos)
    print(f"[gate] student_CE={s_ce:.3f} (<= {args.max_ce}) "
          f"mean_cos={mean_cos:.3f} (>= {args.min_cos}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[gate] FAIL means the warm start is not tracking the teacher. Do not start "
              "training; implement the data-aware SVD/ridge init (paper Alg.1 / App. A.4) "
              "or the LRC co-training/convert path first.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
