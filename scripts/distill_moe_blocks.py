#!/usr/bin/env python3
"""Sequential Layer-Wise Full-Rank Nonlinear MoE-to-MoE Distillation for 4x OLMoE (64 -> 16, top-8).

Distills all 16 layers sequentially (L00 -> L15) against the FULL 64-expert Teacher
on ~524K balanced multi-domain tokens (50% FineWeb+Cosmopedia + 50% open Instruct/QA),
rewriting `gate.weight` (unconstrained) + `gate_up_proj` + `down_proj` at 100% full rank!

Exports a 100% native `OlmoeForCausalLM` (`trust_remote_code=False`) and measures BOTH
Prose Holdout PPL/KL and QA Holdout PPL/KL before & after block distillation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compress_moe_lrc import load_calib_batches
from scripts.explore_4x_math import eval_batches_ce_and_kl, load_sft_calib_batches
from twyd.moe_compress.block_distill import BlockDistillConfig, distill_moe_layer_fullrank
from twyd.moe_compress.pipeline import _advance_hidden_states, _layer_pre_mlp_inputs, _run_layer_forward_sequential


@torch.no_grad()
def compute_teacher_mlp_outputs(
    teacher_mlp: torch.nn.Module,
    x_flat: torch.Tensor,
    device: torch.device,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Run 64-expert teacher `moe_block` on `x_flat [N, H]` in chunks -> CPU bf16 `[N, H]`."""
    N, H = x_flat.shape
    out = torch.empty((N, H), dtype=torch.bfloat16, device="cpu")
    teacher_mlp = teacher_mlp.to(device).eval()
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        xb = x_flat[s:e].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
        yb = teacher_mlp(xb).squeeze(0)
        out[s:e] = yb.to(device="cpu", dtype=torch.bfloat16)
    teacher_mlp.to("cpu")
    torch.cuda.empty_cache()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential Layer-Wise Full-Rank MoE-to-MoE Block Distillation")
    p.add_argument("--teacher", default=os.path.expanduser("~/models/OLMoE-1B-7B-0125-Instruct"))
    p.add_argument("--student", default=os.path.join(ROOT, "output/moe_lrc/olmoe-prog-bal-e16-k8"))
    p.add_argument("--out", default=os.path.join(ROOT, "output/moe_lrc/olmoe-blockdistill-e16-k8"))
    p.add_argument("--num-seqs", type=int, default=1024, help="Total sequences (50% prose + 50% open SFT/QA)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps-per-layer", type=int, default=1200)
    p.add_argument("--batch-tokens", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--router-lr-scale", type=float, default=2.0)
    p.add_argument("--cos-weight", type=float, default=0.2)
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher)

    # Build balanced 50% Prose (FineWeb + Cosmopedia) + 50% Open Instruct/QA (sft_paper8) dataset
    half_seqs = args.num_seqs // 2
    prose_batches = load_calib_batches(
        tok,
        [os.path.join(ROOT, "data/fw_score4_200M.jsonl"), os.path.join(ROOT, "data/pt_cosmopedia_v3_pool.jsonl")],
        num_seqs=half_seqs,
        seq_len=args.seq_len,
        batch_size=4,
    )
    sft_batches = load_sft_calib_batches(
        tok,
        os.path.join(ROOT, "data/sft_paper8.jsonl"),
        num_seqs=half_seqs,
        seq_len=args.seq_len,
        batch_size=4,
        skip_rows=0,
    )
    calib_batches: List[torch.Tensor] = []
    for a, b in zip(prose_batches, sft_batches):
        calib_batches.append(a)
        calib_batches.append(b)
    tot_tokens = sum(b.numel() for b in calib_batches)
    print(f"[distill_blocks] balanced dataset ready: {len(calib_batches)} batches = {tot_tokens:,} tokens", flush=True)

    # Disjoint holdouts for before/after verification
    holdout_prose = load_calib_batches(tok, [os.path.join(ROOT, "data/fineweb_edu_holdout.jsonl")], num_seqs=32, seq_len=512, batch_size=2)
    holdout_qa = load_sft_calib_batches(tok, os.path.join(ROOT, "data/sft_paper8.jsonl"), num_seqs=24, seq_len=512, batch_size=2, skip_rows=50000)

    load_kw = {"dtype": torch.bfloat16, "experts_implementation": "eager", "low_cpu_mem_usage": True}

    print(f"[distill_blocks] loading Teacher to CPU + evaluating holdouts ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, device_map={"": "cpu"}, **load_kw).eval()
    teacher.to(dev)
    t_ce_p, t_ppl_p, _, ref_prose = eval_batches_ce_and_kl(teacher, holdout_prose, args.device)
    t_ce_q, t_ppl_q, _, ref_qa = eval_batches_ce_and_kl(teacher, holdout_qa, args.device)
    print(f"[TEACHER] E=64 k=8 | Prose PPL={t_ppl_p:.2f} | QA PPL={t_ppl_q:.2f}", flush=True)
    teacher.to("cpu")
    torch.cuda.empty_cache()

    print(f"[distill_blocks] loading Student {args.student} ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, device_map={"": "cpu"}, **load_kw).eval()
    student.to(dev)
    b_ce_p, b_ppl_p, b_kl_p, _ = eval_batches_ce_and_kl(student, holdout_prose, args.device, ref_prose)
    b_ce_q, b_ppl_q, b_kl_q, _ = eval_batches_ce_and_kl(student, holdout_qa, args.device, ref_qa)
    print(
        f"[BEFORE]  E={student.config.num_experts} k={student.config.num_experts_per_tok} | "
        f"Prose PPL={b_ppl_p:.2f} KL={b_kl_p:.4f} | QA PPL={b_ppl_q:.2f} KL={b_kl_q:.4f} | SumKL={b_kl_p+b_kl_q:.4f}",
        flush=True,
    )

    # Sequential hidden-state propagation (embed once, then layer 0 -> 15 on GPU one layer at a time!)
    h_stu, pe_list, pid_list = _run_layer_forward_sequential(student, calib_batches, dev)
    student.to("cpu")
    torch.cuda.empty_cache()

    cfg = BlockDistillConfig(
        steps_per_layer=args.steps_per_layer,
        batch_tokens=args.batch_tokens,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
        cos_weight=args.cos_weight,
        aux_weight=args.aux_weight,
    )

    layer_stats = []
    n_layers = len(student.model.layers)
    for l_idx in range(n_layers):
        s_layer = student.model.layers[l_idx].to(dev).eval()
        # 1. Capture exact pre-MLP inputs entering layer l from previous distilled layers
        x_flat = _layer_pre_mlp_inputs(s_layer, h_stu, pe_list, pid_list, dev)
        s_layer.to("cpu")
        torch.cuda.empty_cache()

        # 2. Compute exact 64-expert Teacher MoE output vector Y_l^(T)(x_flat)
        y_teacher = compute_teacher_mlp_outputs(teacher.model.layers[l_idx].mlp, x_flat, dev)

        # 3. Full-rank nonlinear MoE-to-MoE distillation on layer l's student MLP
        s_mlp = student.model.layers[l_idx].mlp.to(dev)
        st = distill_moe_layer_fullrank(s_mlp, x_flat, y_teacher, cfg, layer_idx=l_idx, device=dev, verbose=True)
        layer_stats.append(asdict(st))
        s_mlp.to("cpu")
        del x_flat, y_teacher
        torch.cuda.empty_cache()

        # 4. Advance student hidden states through the newly distilled layer l
        s_layer = student.model.layers[l_idx].to(dev).eval()
        h_stu = _advance_hidden_states(s_layer, h_stu, pe_list, pid_list, dev)
        s_layer.to("cpu")
        torch.cuda.empty_cache()

    del h_stu, pe_list, pid_list, teacher
    torch.cuda.empty_cache()

    # Evaluate final 16-layer block-distilled native model
    student.to(dev).eval()
    a_ce_p, a_ppl_p, a_kl_p, _ = eval_batches_ce_and_kl(student, holdout_prose, args.device, ref_prose)
    a_ce_q, a_ppl_q, a_kl_q, _ = eval_batches_ce_and_kl(student, holdout_qa, args.device, ref_qa)
    print(
        f"\n[AFTER]   E={student.config.num_experts} k={student.config.num_experts_per_tok} | "
        f"Prose PPL={a_ppl_p:.2f} KL={a_kl_p:.4f} | QA PPL={a_ppl_q:.2f} KL={a_kl_q:.4f} | SumKL={a_kl_p+a_kl_q:.4f}",
        flush=True,
    )

    student.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    report = {
        "config": asdict(cfg),
        "tokens": tot_tokens,
        "teacher": {"prose_ppl": t_ppl_p, "qa_ppl": t_ppl_q},
        "before": {"prose_ppl": b_ppl_p, "prose_kl": b_kl_p, "qa_ppl": b_ppl_q, "qa_kl": b_kl_q, "sum_kl": b_kl_p + b_kl_q},
        "after": {"prose_ppl": a_ppl_p, "prose_kl": a_kl_p, "qa_ppl": a_ppl_q, "qa_kl": a_kl_q, "sum_kl": a_kl_p + a_kl_q},
        "mean_init_val_mse": sum(s["init_val_rel_mse"] for s in layer_stats) / len(layer_stats),
        "mean_best_val_mse": sum(s["best_val_rel_mse"] for s in layer_stats) / len(layer_stats),
        "total_elapsed_min": (time.time() - t_total) / 60.0,
        "layers": layer_stats,
    }
    json.dump(report, open(os.path.join(args.out, "block_distill_report.json"), "w"), indent=2)
    print(
        f"[distill_blocks] DONE in {report['total_elapsed_min']:.1f} min -> {args.out} | "
        f"mean valMSE {report['mean_init_val_mse']:.4f} -> {report['mean_best_val_mse']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
