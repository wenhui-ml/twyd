#!/usr/bin/env python3
"""Stage 2 Online KD + Cluster-Router-KL Recovery for LRC-compressed native MoE models.

Runs frozen MoE Teacher on `--teacher-device` (e.g. cuda:1) and trainable native
HF MoE Student on `--student-device` (e.g. cuda:0) with CPU-offloaded fp32 AdamW,
Fused Chunked Top-256 KL + CE, and Per-Layer Cluster-Aggregated Router-KL.

Outputs 100% standard native HF MoE checkpoints (`student.save_pretrained`, zero
trust_remote_code).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Iterator, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from twyd.moe_compress.recover import (
    CpuOffloadedAdamW,
    RecoveryConfig,
    aggregate_teacher_router_targets,
    compute_fused_logit_kl_ce,
    compute_router_kl_loss,
    extract_teacher_topk_targets,
    inject_moe_expert_lora_,
    load_cluster_maps,
    save_merged_native_moe,
    select_trainable_params,
)


def stream_packed_sequences(
    tokenizer,
    data_files: List[str],
    seq_len: int,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Endless stream of packed fixed-length token sequences (`[seq_len]`)."""
    eos = tokenizer.eos_token_id
    buf: List[int] = []
    epoch = 0
    while True:
        epoch += 1
        for path in data_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                    if not text:
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    buf.extend(ids)
                    if eos is not None:
                        buf.append(int(eos))
                    while len(buf) >= seq_len:
                        yield torch.tensor(buf[:seq_len], dtype=torch.long)
                        buf = buf[seq_len:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 Online KD + Router-KL MoE Recovery")
    p.add_argument("--teacher", required=True, help="Teacher MoE model dir")
    p.add_argument("--student", required=True, help="Step-0 LRC-compressed native MoE dir")
    p.add_argument("--out", required=True, help="Output checkpoint directory")
    p.add_argument("--cluster-map", default=None, help="Path to lrc_cluster_map.json (defaults to <student>/lrc_cluster_map.json)")
    p.add_argument("--data", nargs="+", required=True, help="JSONL training text files")
    p.add_argument("--teacher-device", default="cuda:1")
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--router-lr-scale", type=float, default=0.5)
    p.add_argument("--lora-rank", type=int, default=64, help=">0: mergeable per-expert LoRA rank; 0: full-rank")
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--kl-weight", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--router-kl-weight", type=float, default=0.2)
    p.add_argument("--router-target-mode", default="survivor", choices=["survivor", "cluster"])
    p.add_argument("--aux-loss-weight", type=float, default=0.01)
    p.add_argument("--kl-topk", type=int, default=256)
    p.add_argument("--kl-temp", type=float, default=1.0)
    p.add_argument("--trainable-scope", default="moe", choices=["moe", "all"])
    p.add_argument("--cpu-offload", dest="cpu_offload", action="store_true", default=False)
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    p.add_argument("--grad-checkpoint", dest="grad_ckpt", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", dest="grad_ckpt", action="store_false")
    return p.parse_args()


def _lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    prog = float(step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_dev = torch.device(args.teacher_device)
    s_dev = torch.device(args.student_device)
    cmap_path = args.cluster_map or os.path.join(args.student, "lrc_cluster_map.json")

    new_e, cluster_maps, survivor_maps = load_cluster_maps(cmap_path)
    print(f"[recover] loaded cluster_map ({len(cluster_maps)} layers, K'={new_e}, mode={args.router_target_mode})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    load_kw = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if getattr(torch.version, "hip", None):
        load_kw["experts_implementation"] = "eager"
    print(f"[recover] loading teacher -> {t_dev} ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, device_map={"": str(t_dev)}, **load_kw)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[recover] loading student -> {s_dev} ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, device_map={"": str(s_dev)}, **load_kw)
    student.config.use_cache = False
    if args.lora_rank > 0:
        n_wrapped = inject_moe_expert_lora_(student, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f"[recover] injected mergeable MoeExpertLoRA (r={args.lora_rank}, alpha={args.lora_alpha}) on {n_wrapped} layers", flush=True)
    if args.grad_ckpt:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cfg = RecoveryConfig(
        kl_temperature=args.kl_temp,
        kl_topk=args.kl_topk,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        router_kl_weight=args.router_kl_weight,
        router_target_mode=args.router_target_mode,
        aux_loss_weight=args.aux_loss_weight,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        trainable_scope=args.trainable_scope,
        cpu_offload_optimizer=args.cpu_offload,
    )
    groups = select_trainable_params(student, cfg)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_total = sum(p.numel() for p in student.parameters())
    print(
        f"[recover] trainable params: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
        f"({100*n_train/n_total:.1f}%), cpu_offload={cfg.cpu_offload_optimizer}",
        flush=True,
    )

    base_lrs = [g["lr"] for g in groups]
    if cfg.cpu_offload_optimizer:
        opt = CpuOffloadedAdamW(groups)
    else:
        opt = torch.optim.AdamW(groups)

    seq_iter = stream_packed_sequences(tokenizer, args.data, args.seq_len)
    student.train()
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        t_step = time.time()
        opt.zero_grad()

        # Cosine LR schedule
        mult = _lr_factor(step - 1, args.warmup_steps, args.max_steps)
        for g, blr in zip(opt.param_groups, base_lrs):
            g["lr"] = blr * mult

        acc_loss = acc_kl = acc_ce = acc_rkl = acc_aux = 0.0

        for _ in range(args.grad_accum):
            batch_cpu = torch.stack([next(seq_iter) for _ in range(args.batch_size)], dim=0)
            ids_t = batch_cpu.to(device=t_dev, non_blocking=True)
            ids_s = batch_cpu.to(device=s_dev, non_blocking=True)

            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids_t,
                    use_cache=False,
                    output_router_logits=(cfg.router_kl_weight > 0),
                )
                top_idx, logp_t_top, p_t_top, p_t_tail = extract_teacher_topk_targets(
                    t_out.logits, top_k=cfg.kl_topk, T=cfg.kl_temperature, chunk_size=cfg.chunk_size
                )
                t_router_targets = (
                    aggregate_teacher_router_targets(
                        t_out.router_logits,
                        cluster_maps,
                        new_e,
                        survivors=survivor_maps,
                        mode=cfg.router_target_mode,
                    )
                    if (cfg.router_kl_weight > 0 and t_out.router_logits is not None)
                    else []
                )
                del t_out

            top_idx_s = top_idx.to(s_dev, non_blocking=True)
            logp_s = logp_t_top.to(s_dev, non_blocking=True)
            pt_s = p_t_top.to(s_dev, non_blocking=True)
            ptail_s = p_t_tail.to(s_dev, non_blocking=True)

            s_out = student(
                input_ids=ids_s,
                use_cache=False,
                output_router_logits=(cfg.router_kl_weight > 0 or cfg.aux_loss_weight > 0),
            )
            kl_l, ce_l = compute_fused_logit_kl_ce(
                s_out.logits,
                ids_s,
                top_idx_s,
                logp_s,
                pt_s,
                ptail_s,
                T=cfg.kl_temperature,
                chunk_size=cfg.chunk_size,
                use_checkpoint=args.grad_ckpt,
            )
            rkl_l = (
                compute_router_kl_loss(s_out.router_logits, t_router_targets)
                if (cfg.router_kl_weight > 0 and s_out.router_logits is not None)
                else torch.zeros((), device=s_dev)
            )
            aux_l = (
                s_out.aux_loss.float()
                if (cfg.aux_loss_weight > 0 and getattr(s_out, "aux_loss", None) is not None)
                else torch.zeros((), device=s_dev)
            )

            loss = (
                cfg.kl_weight * kl_l
                + cfg.ce_weight * ce_l
                + cfg.router_kl_weight * rkl_l
                + cfg.aux_loss_weight * aux_l
            )
            (loss / args.grad_accum).backward()

            acc_loss += float(loss.item()) / args.grad_accum
            acc_kl += float(kl_l.item()) / args.grad_accum
            acc_ce += float(ce_l.item()) / args.grad_accum
            acc_rkl += float(rkl_l.item()) / args.grad_accum
            acc_aux += float(aux_l.item()) / args.grad_accum
            del s_out, loss, kl_l, ce_l, rkl_l, aux_l

        if cfg.cpu_offload_optimizer:
            gnorm = opt.step(max_grad_norm=cfg.max_grad_norm)
        else:
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], cfg.max_grad_norm
            ).item())
            opt.step()

        if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
            dt = time.time() - t_step
            toks_step = args.batch_size * args.grad_accum * args.seq_len
            vram_s = torch.cuda.max_memory_allocated(s_dev) / (1024 ** 3) if s_dev.type == "cuda" else 0.0
            vram_t = torch.cuda.max_memory_allocated(t_dev) / (1024 ** 3) if t_dev.type == "cuda" else 0.0
            print(
                f"[recover] step {step:4d}/{args.max_steps} | loss={acc_loss:.4f} "
                f"(kl={acc_kl:.4f} ce={acc_ce:.4f} rkl={acc_rkl:.4f} aux={acc_aux:.4f}) | "
                f"gnorm={gnorm:.2f} lr={opt.param_groups[0]['lr']:.2e} | "
                f"{dt:.1f}s ({toks_step/max(dt,1e-3):.0f} tok/s) | vram S={vram_s:.1f}G T={vram_t:.1f}G",
                flush=True,
            )

        if step % args.save_steps == 0 or step == args.max_steps:
            ck_dir = os.path.join(args.out, f"checkpoint-{step}")
            save_merged_native_moe(student, ck_dir, tokenizer=tokenizer)
            print(f"[recover] saved merged 100% native HF MoE checkpoint -> {ck_dir}", flush=True)

    print(f"[recover] DONE in {(time.time()-t_start)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
