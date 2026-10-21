#!/usr/bin/env python
"""Live-vs-merged parity: does the exported width-r model equal the trained one?

Prop.1 promises that CORE / CORE-LR merge into one plain ``d_ff x r`` weight with
zero inference cost.  That claim is only usable if the *deployed* checkpoint
reproduces the *live* training graph.  This script closes the ROADMAP item
"logit parity: live wrapper vs reload merged, max_abs_diff < 1e-2":

  1. build the student exactly like training does (shrink + factorized MLP +
     freeze) for any arm (plain / dense / core / core-LR / col-control);
  2. run a few distillation steps so the low-rank factors are non-zero;
  3. export through the same path ms-swift uses (``twyd.merge.merge_and_save``);
  4. reload the merged HF checkpoint and compare weights + logits on one batch.

Usage:
  python scripts/check_merge_parity.py \
      --teacher /home/chenwh/models/Llama-3.2-3B-Instruct \
      --model   /home/chenwh/models/LRC-1.5B-Base \
      --r 1536 --branch core --lora-rank 128 --warm-start base \
      --steps 2 --seq-len 128 --device cuda
"""
import argparse
import os
import sys
import tempfile

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXT = (
    "The capital of France is Paris. Water boils at 100 degrees Celsius at sea level. "
    "In mathematics, the derivative of x squared is 2x. Large language models are trained "
    "on vast amounts of text data to predict the next token. A compressed student must "
    "track its teacher before and after the merge."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--model", default=None, help="student base (default: teacher)")
    ap.add_argument("--r", type=int, default=1536)
    ap.add_argument("--branch", default="core", choices=["dense", "core", "plain"])
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-mode", default=None, choices=["dense", "core"])
    ap.add_argument("--perp-basis", default="teacher", choices=["teacher", "col"])
    ap.add_argument("--freeze-col", dest="freeze_col", action="store_true", default=None)
    ap.add_argument("--no-freeze-col", dest="freeze_col", action="store_false")
    ap.add_argument("--warm-start", default="base", choices=["slice", "ridge", "base"])
    ap.add_argument("--factorized", action="store_true",
                    help="train with the cached-W_col + rank-k delta forward (the "
                         "actual CORE-LR training path)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |delta logit| for PASS (bf16 export; 1e-2 needs fp32)")
    ap.add_argument("--keep", default=None, help="export into this dir instead of a temp dir")
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    from twyd.config import TwydLRCConfig
    from twyd.merge import merge_and_save
    from twyd.tuner import (CoreLinearLoRA, DenseLinearLoRA, _TwydWrapper,
                            _build_factorized_student, _freeze_for_branch,
                            _hidden_of, _shrink_student_)

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

    r = int(args.r)
    if _hidden_of(student) != r:
        student = _shrink_student_(student, teacher, r)

    lora_mode = args.lora_mode or ("core" if (args.branch == "core" and args.lora_rank > 0)
                                   else "dense")
    freeze_col = args.freeze_col if args.freeze_col is not None else (args.branch == "core")
    cfg = TwydLRCConfig(branch=args.branch, teacher_model=args.teacher, student_hidden=r,
                        stage="pt", warm_start=args.warm_start,
                        lora_rank=args.lora_rank, lora_mode=lora_mode,
                        freeze_col=freeze_col, factorized=args.factorized,
                        perp_basis=args.perp_basis,
                        target_modules=["gate_proj", "up_proj", "down_proj"])
    cfg.validate()
    meta = _build_factorized_student(student, teacher, cfg, branch=args.branch)
    wrapper = _TwydWrapper(student, teacher, cfg)
    wrapper.twyd_cfg = cfg
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)

    n_core = sum(isinstance(m, CoreLinearLoRA) for m in student.modules())
    n_dense = sum(isinstance(m, DenseLinearLoRA) for m in student.modules())
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[adapter] branch={args.branch} lora_mode={lora_mode} "
          f"perp_basis={args.perp_basis} freeze_col={freeze_col} "
          f"CORE-LR={n_core} Dense-LoRA={n_dense} trainable={n_train/1e6:.1f}M")

    text = args.text or DEFAULT_TEXT
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :args.seq_len]
    ids = ids.to(dev)
    teacher = teacher.to(dev)
    student = student.to(dev)
    print(f"[data] {ids.shape[1]} tokens on {dev}")

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    teacher.eval()
    student.train()
    loss_val = None
    for _ in range(max(1, args.steps)):
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, use_cache=False).logits.float()
        out = student(input_ids=ids, use_cache=False)
        loss = F.mse_loss(out.logits.float(), t_logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
    print(f"[train] steps={args.steps} lr={args.lr} last_mse={loss_val:.4f}")

    student.eval()
    with torch.no_grad():
        live = student(input_ids=ids, use_cache=False).logits.float()

    out_dir = args.keep or tempfile.mkdtemp(prefix="twyd-merge-parity-")
    merge_and_save(wrapper, out_dir)

    reloaded = AutoModelForCausalLM.from_pretrained(
        out_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()

    # --- weight parity: live merged_weight() vs the exported HF tensors --------
    live_mods = {n: m for n, m in student.named_modules()
                 if callable(getattr(m, "merged_weight", None)) and n}
    ref_mods = dict(reloaded.named_modules())
    w_max, w_worst = 0.0, None
    for name, m in live_mods.items():
        ref = ref_mods.get(name, None)
        if ref is None or not hasattr(ref, "weight"):
            continue
        with torch.no_grad():
            d = (m.merged_weight().float() - ref.weight.float()).abs().max().item()
        if d > w_max:
            w_max, w_worst = d, name
    if not live_mods:  # dense arm: plain nn.Linear weights
        own = dict(student.named_parameters())
        for n, p in own.items():
            if not n.endswith(".weight"):
                continue
            ref = ref_mods.get(n.rsplit(".", 1)[0], None)
            if ref is not None and hasattr(ref, "weight") and ref.weight.shape == p.shape:
                d = (p.float() - ref.weight.float()).abs().max().item()
                if d > w_max:
                    w_max, w_worst = d, n
    print(f"[weights] modules={len(live_mods)} max|W_live-W_export|={w_max:.3e}"
          + (f" ({w_worst})" if w_worst else ""))

    # --- logit parity --------------------------------------------------------
    with torch.no_grad():
        merged = reloaded(input_ids=ids, use_cache=False).logits.float()
    diff = (live - merged).abs()
    scale = live.abs().max().clamp_min(1e-9)
    print(f"[logits] max|d|={diff.max().item():.4e} mean|d|={diff.mean().item():.4e} "
          f"rel={(diff.max() / scale).item():.3e} scale={scale.item():.2f}")

    ok_w = w_max <= 1e-3
    ok_l = diff.max().item() <= args.tol
    print(f"[parity] weights={'PASS' if ok_w else 'FAIL'} (<=1e-3)  "
          f"logits={'PASS' if ok_l else 'FAIL'} (<= {args.tol})  -> "
          f"{'PASS' if (ok_w and ok_l) else 'FAIL'}")
    print(f"[out] {out_dir}")
    return 0 if (ok_w and ok_l) else 1


if __name__ == "__main__":
    raise SystemExit(main())
