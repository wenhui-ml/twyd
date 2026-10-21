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
    ap.add_argument("--freeze-col", dest="freeze_col", action="store_true", default=None,
                    help="freeze the plain-LRC col path (CORE-LR; default for --branch core, "
                         "mirrors TWYD_FREEZE_COL=true in train_core_lr_pilot.sh)")
    ap.add_argument("--no-freeze-col", dest="freeze_col", action="store_false",
                    help="paper-exact full CORE: train Z_col together with Z_perp")
    ap.add_argument("--lora-mode", default=None, choices=["dense", "core"],
                    help="where the low-rank adapter lives: 'core' = CORE-LR "
                         "(rank-k complement, what train_core_lr_pilot.sh builds), "
                         "'dense' = Dense-LoRA. Default: core when "
                         "--branch core and --lora-rank > 0, else dense.")
    ap.add_argument("--warm-start", default="slice", choices=["slice", "ridge", "base"],
                    help="slice / ridge / base (official width-r LRC checkpoint)")
    ap.add_argument("--master-p", default="none", choices=["none", "embed_svd"],
                    help="project embedding/attention/MLP onto embedding-SVD P")
    ap.add_argument("--head-mode", default="select", choices=["select", "merge"],
                    help="P attention head mapping: select first heads or average groups")
    ap.add_argument("--ridge-perp", default="residual", choices=["zero", "residual", "lowrank"])
    ap.add_argument("--rank-map", default="", help="JSON dict/path: per-module CORE-LR k")
    ap.add_argument("--perp-basis", default="teacher", choices=["teacher", "col"],
                    help="teacher = true complement (CORE/CORE-LR); col = P6 "
                         "equal-parameter control that writes the same number of "
                         "coordinates back inside col(T) (paper §6.1)")
    ap.add_argument("--factorized", action="store_true",
                    help="use cached W_col + rank-k delta forward")
    ap.add_argument("--calib-ids", default=None, help=".pt/.npy/.jsonl/.txt for --warm-start ridge")
    ap.add_argument("--calib-seq-len", type=int, default=256)
    ap.add_argument("--calib-num-seqs", type=int, default=32)
    ap.add_argument("--calib-attn", action="store_true",
                    help="also ridge q/k/v/o with head-projected teacher targets")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ce", type=float, default=6.0)
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--min-tok-acc", type=float, default=0.20,
                    help="absolute floor on student token accuracy")
    ap.add_argument("--min-agree", type=float, default=0.30,
                    help="min top-1 agreement with teacher (basis-free)")
    ap.add_argument("--min-acc-ratio", type=float, default=0.50,
                    help="student tok_acc must be >= this fraction of teacher's")
    ap.add_argument("--cos-check", action="store_true",
                    help="also gate on hidden cosine; only valid for slice-type "
                         "inits (a trained LRC base uses a different residual "
                         "basis and scale, so cosine is meaningless there)")
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    from twyd.config import TwydLRCConfig
    from twyd.init import (build_master_P, calibrate_attention,
                           project_student_with_P_)
    from twyd.tuner import (DenseLinearLoRA, CoreLinearLoRA, _TwydWrapper,
                            _build_factorized_student,
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
    # The adapter placement decides what actually trains: --branch core with
    # --lora-rank > 0 must be CORE-LR (rank-k complement), not Dense-LoRA.
    # Before this flag existed, lora_mode silently stayed "dense", so the gate
    # validated a different module than train_core_lr_pilot.sh builds.
    lora_mode = args.lora_mode or ("core" if (args.branch == "core" and args.lora_rank > 0)
                                   else "dense")
    freeze_col = args.freeze_col if args.freeze_col is not None else (args.branch == "core")
    cfg = TwydLRCConfig(branch=args.branch, teacher_model=args.teacher, student_hidden=r,
                        stage="pt", warm_start=args.warm_start,
                        ridge_perp=args.ridge_perp, calib_ids=args.calib_ids,
                        calib_seq_len=args.calib_seq_len,
                        calib_num_seqs=args.calib_num_seqs,
                        lora_rank=args.lora_rank, lora_mode=lora_mode,
                        freeze_col=freeze_col,
                        rank_map=args.rank_map,
                        factorized=args.factorized,
                        perp_basis=args.perp_basis,
                        target_modules=["gate_proj", "up_proj", "down_proj"])
    cfg.validate()
    meta = _build_factorized_student(student, teacher, cfg, branch=args.branch)
    n_core_lr = sum(isinstance(m, CoreLinearLoRA) for m in student.modules())
    n_dense_lora = sum(isinstance(m, DenseLinearLoRA) for m in student.modules())
    print(f"[adapter] branch={args.branch} lora_mode={lora_mode} "
          f"perp_basis={args.perp_basis} "
          f"k={args.lora_rank} -> CORE-LR modules={n_core_lr}, "
          f"Dense-LoRA modules={n_dense_lora}")
    if args.branch == "core" and args.lora_rank > 0 and lora_mode == "core" and n_core_lr == 0:
        print("[adapter] FAIL: CORE-LR requested but no CoreLinearLoRA was built "
              "(check target_modules / rank / freeze_col).")
        return 1
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

    # Basis-free agreement: top-1 token match + symmetric KL on a common
    # vocabulary slice.  This is what matters for a trained LRC base, whose
    # residual basis and scale differ from the teacher's.
    agree = kl_s = kl_t = None
    try:
        tl = t_out.logits.float()[..., :-1, :]
        sl = s_out.logits.float()[..., :-1, :]
        V = min(tl.size(-1), sl.size(-1))
        tl, sl = tl[..., :V], sl[..., :V]
        agree = (tl.argmax(-1) == sl.argmax(-1)).float().mean().item()
        logp_t = F.log_softmax(tl, dim=-1)
        logp_s = F.log_softmax(sl, dim=-1)
        kl_st = F.kl_div(logp_s, logp_t, log_target=True, reduction="batchmean").item()
        kl_ts = F.kl_div(logp_t, logp_s, log_target=True, reduction="batchmean").item()
        kl_s = kl_st
        kl_t = kl_ts
        print(f"[logits] top1_agree={agree:.4f} KL(s||t)={kl_st:.4f} KL(t||s)={kl_ts:.4f}")
    except Exception as e:
        print(f"[logits] comparison skipped: {e}")

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

    ok = (s_ce <= args.max_ce) and (s_acc >= args.min_tok_acc)
    if agree is not None:
        ok = ok and (agree >= args.min_agree)
    if t_acc > 0:
        ok = ok and (s_acc >= args.min_acc_ratio * t_acc)
    if args.cos_check:
        ok = ok and (mean_cos >= args.min_cos)
    print(f"[gate] student_CE={s_ce:.3f} (<= {args.max_ce}) "
          f"tok_acc={s_acc:.4f} (>= {args.min_tok_acc}, "
          f">= {args.min_acc_ratio:.2f}x teacher={t_acc:.4f})"
          + (f" agree={agree:.3f} (>= {args.min_agree})" if agree is not None else "")
          + (f" mean_cos={mean_cos:.3f} (>= {args.min_cos})" if args.cos_check else
             f" mean_cos={mean_cos:.3f} (informational)")
          + f" -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[gate] FAIL: the warm start is not tracking the teacher. Do not start "
              "training; use an official LRC/PT width-r base (--warm-start base) or "
              "implement the data-aware SVD/ridge init (paper Alg.1 / App. A.4).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
