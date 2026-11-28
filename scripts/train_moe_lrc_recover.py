#!/usr/bin/env python3
"""Stage 2 Online KD + Hidden Alignment + Router-KL Recovery for LRC-compressed native MoE models.

Supports BOTH:
  - `--mode pt`  : Packed pretraining/textbook recovery (Cosmopedia + FineWeb-Edu)
  - `--mode sft` : Multi-turn Instruct/QA Online KD-SFT (masks prompt/pad tokens to -100,
                   supervising strictly on assistant response tokens with online 64-expert Teacher)

Runs frozen MoE Teacher on `--teacher-device` (cuda:1) and trainable native HF MoE Student
on `--student-device` (cuda:0) with either on-GPU AdamW (mergeable `MoeExpertLoRA`) or
`CpuOffloadedAdamW` (full-rank 1.88B/2.09B).

Outputs 100% standard native HF MoE checkpoints (`trust_remote_code=False`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Iterator, List, Tuple

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
    compute_hidden_cosine_loss,
    compute_router_kl_loss,
    extract_teacher_hidden_targets,
    extract_teacher_topk_targets,
    inject_moe_expert_lora_,
    load_cluster_maps,
    save_merged_native_moe,
    select_trainable_params,
)


def stream_packed_pt_batches(
    tokenizer,
    data_files: List[str],
    seq_len: int,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Endless stream of packed PT `(input_ids, labels, attention_mask)` batches."""
    eos = tokenizer.eos_token_id
    buf: List[int] = []
    seqs: List[torch.Tensor] = []
    while True:
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
                        seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                        buf = buf[seq_len:]
                        if len(seqs) == batch_size:
                            ids_b = torch.stack(seqs, dim=0)
                            mask_b = torch.ones_like(ids_b)
                            yield ids_b, ids_b.clone(), mask_b
                            seqs = []


def encode_sft_messages(
    tokenizer,
    messages: List[dict],
    max_len: int,
) -> Tuple[List[int], List[int]]:
    """Encode multi-turn `messages` with assistant-only loss mask (`-100` on prompt turns)."""
    ids: List[int] = []
    labels: List[int] = []
    has_template = bool(getattr(tokenizer, "chat_template", None))

    # Build turn-by-turn prefixes so assistant spans are identified with 100% exactness
    clean_msgs = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
        for m in messages
        if m.get("role") in ("system", "user", "assistant") and str(m.get("content", "")).strip()
    ]
    if not clean_msgs:
        return [], []

    # Drop trailing user turns that have no assistant answer
    while clean_msgs and clean_msgs[-1]["role"] != "assistant":
        clean_msgs.pop()
    if not clean_msgs:
        return [], []

    if has_template:
        prev_len = 0
        for i in range(len(clean_msgs)):
            role = clean_msgs[i]["role"]
            if role == "assistant":
                prompt_text = tokenizer.apply_chat_template(
                    clean_msgs[:i], tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    clean_msgs[: i + 1], tokenize=False, add_generation_prompt=False
                )
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                # Append prompt delta (masked) + assistant delta (supervised)
                p_delta = prompt_ids[prev_len:]
                a_delta = full_ids[len(prompt_ids):]
                ids.extend(p_delta)
                labels.extend([-100] * len(p_delta))
                ids.extend(a_delta)
                labels.extend(a_delta)
                prev_len = len(full_ids)
                if len(ids) >= max_len:
                    break
    else:
        for m in clean_msgs:
            role = m["role"]
            chunk = tokenizer.encode(f"<{role}>\n{m['content']}\n", add_special_tokens=False)
            ids.extend(chunk)
            labels.extend(chunk if role == "assistant" else [-100] * len(chunk))
            if len(ids) >= max_len:
                break

    ids = ids[:max_len]
    labels = labels[:max_len]
    if not any(y != -100 for y in labels[1:]):
        return [], []
    return ids, labels


def stream_packed_sft_batches(
    tokenizer,
    data_files: List[str],
    seq_len: int,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Endless stream of padded/packed SFT `(input_ids, labels, attention_mask)` batches.

    Packs multiple conversations up to `seq_len` (padded with `pad_token_id` and
    `labels=-100` where needed) so GPU utilization stays high while preserving
    strict assistant-only supervision.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id
    cur_ids: List[int] = []
    cur_lbl: List[int] = []
    batch_ids: List[torch.Tensor] = []
    batch_lbl: List[torch.Tensor] = []
    batch_msk: List[torch.Tensor] = []

    while True:
        for path in data_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    msgs = obj.get("messages")
                    if not msgs:
                        continue
                    s_ids, s_lbl = encode_sft_messages(tokenizer, msgs, max_len=seq_len)
                    if not s_ids:
                        continue
                    if cur_ids and len(cur_ids) + len(s_ids) > seq_len:
                        pad_n = seq_len - len(cur_ids)
                        batch_ids.append(torch.tensor(cur_ids + [pad_id] * pad_n, dtype=torch.long))
                        batch_lbl.append(torch.tensor(cur_lbl + [-100] * pad_n, dtype=torch.long))
                        batch_msk.append(torch.tensor([1] * len(cur_ids) + [0] * pad_n, dtype=torch.long))
                        cur_ids, cur_lbl = [], []
                        if len(batch_ids) == batch_size:
                            yield (
                                torch.stack(batch_ids, dim=0),
                                torch.stack(batch_lbl, dim=0),
                                torch.stack(batch_msk, dim=0),
                            )
                            batch_ids, batch_lbl, batch_msk = [], [], []
                    cur_ids.extend(s_ids)
                    cur_lbl.extend(s_lbl)
                    if eos_id is not None and len(cur_ids) < seq_len and (not cur_ids or cur_ids[-1] != eos_id):
                        cur_ids.append(int(eos_id))
                        cur_lbl.append(-100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 Online KD + Hidden + Router-KL MoE Recovery")
    p.add_argument("--teacher", required=True, help="Teacher MoE model dir")
    p.add_argument("--student", required=True, help="Step-0 or resumed native MoE dir")
    p.add_argument("--out", required=True, help="Output checkpoint directory")
    p.add_argument("--cluster-map", default=None, help="Path to lrc_cluster_map.json (optional if router-kl=0)")
    p.add_argument("--data", nargs="+", required=True, help="JSONL training files")
    p.add_argument("--mode", default="pt", choices=["pt", "sft"])
    p.add_argument("--teacher-device", default="cuda:1")
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--save-steps", type=int, default=300)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--router-lr-scale", type=float, default=0.5)
    p.add_argument("--attn-lr-scale", type=float, default=0.3)
    p.add_argument("--lora-rank", type=int, default=64, help=">0: mergeable per-expert LoRA; 0: full-rank")
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--kl-weight", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--jsd-beta", type=float, default=0.0, help="0=Forward KL; (0,1)=Generalized JSD")
    p.add_argument("--router-kl-weight", type=float, default=0.2)
    p.add_argument("--router-target-mode", default="survivor", choices=["survivor", "cluster"])
    p.add_argument("--hidden-align-weight", type=float, default=0.5)
    p.add_argument("--hidden-align-layers", type=str, default="3,7,11,15")
    p.add_argument("--aux-loss-weight", type=float, default=0.01)
    p.add_argument("--kl-topk", type=int, default=256)
    p.add_argument("--kl-temp", type=float, default=1.0)
    p.add_argument("--trainable-scope", default="all_no_embed", choices=["moe", "all_no_embed", "all"])
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
    new_e = 0
    cluster_maps = {}
    survivor_maps = {}
    if args.router_kl_weight > 0 and os.path.isfile(cmap_path):
        new_e, cluster_maps, survivor_maps = load_cluster_maps(cmap_path)
        print(f"[recover] loaded cluster_map ({len(cluster_maps)} layers, K'={new_e}, mode={args.router_target_mode})", flush=True)
    elif args.router_kl_weight > 0:
        print(f"[recover] WARN: cluster_map not found at {cmap_path}; disabling router_kl_weight", flush=True)
        args.router_kl_weight = 0.0

    hidden_layers = tuple(int(x.strip()) for x in args.hidden_align_layers.split(",") if x.strip())

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
        jsd_beta=args.jsd_beta,
        router_kl_weight=args.router_kl_weight,
        router_target_mode=args.router_target_mode,
        hidden_align_weight=args.hidden_align_weight,
        hidden_align_layers=hidden_layers,
        aux_loss_weight=args.aux_loss_weight,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
        attn_lr_scale=args.attn_lr_scale,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        trainable_scope=args.trainable_scope,
        cpu_offload_optimizer=args.cpu_offload,
    )
    groups = select_trainable_params(student, cfg)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_total = sum(p.numel() for p in student.parameters())
    print(
        f"[recover] mode={args.mode} scope={cfg.trainable_scope} trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
        f"({100*n_train/n_total:.1f}%), lora_r={cfg.lora_rank}, cpu_offload={cfg.cpu_offload_optimizer}, "
        f"h_align={cfg.hidden_align_weight}@{hidden_layers}, jsd_beta={cfg.jsd_beta}",
        flush=True,
    )

    base_lrs = [g["lr"] for g in groups]
    opt = CpuOffloadedAdamW(groups) if cfg.cpu_offload_optimizer else torch.optim.AdamW(groups)

    batch_iter = (
        stream_packed_sft_batches(tokenizer, args.data, args.seq_len, args.batch_size)
        if args.mode == "sft"
        else stream_packed_pt_batches(tokenizer, args.data, args.seq_len, args.batch_size)
    )

    student.train()
    t_start = time.time()
    need_hidden = cfg.hidden_align_weight > 0 and len(hidden_layers) > 0
    need_router_t = cfg.router_kl_weight > 0 and len(cluster_maps) > 0

    for step in range(1, args.max_steps + 1):
        t_step = time.time()
        opt.zero_grad()

        mult = _lr_factor(step - 1, args.warmup_steps, args.max_steps)
        for g, blr in zip(opt.param_groups, base_lrs):
            g["lr"] = blr * mult

        acc_loss = acc_kl = acc_ce = acc_rkl = acc_hid = acc_aux = 0.0

        for _ in range(args.grad_accum):
            ids_cpu, lbl_cpu, msk_cpu = next(batch_iter)
            ids_t = ids_cpu.to(device=t_dev, non_blocking=True)
            msk_t = msk_cpu.to(device=t_dev, non_blocking=True)
            ids_s = ids_cpu.to(device=s_dev, non_blocking=True)
            lbl_s = lbl_cpu.to(device=s_dev, non_blocking=True)
            msk_s = msk_cpu.to(device=s_dev, non_blocking=True)

            with torch.no_grad():
                t_out = teacher(
                    input_ids=ids_t,
                    attention_mask=msk_t if args.mode == "sft" else None,
                    use_cache=False,
                    output_router_logits=need_router_t,
                    output_hidden_states=need_hidden,
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
                    if (need_router_t and t_out.router_logits is not None)
                    else []
                )
                t_hidden_targets = (
                    extract_teacher_hidden_targets(t_out.hidden_states, hidden_layers)
                    if (need_hidden and t_out.hidden_states is not None)
                    else []
                )
                del t_out

            top_idx_s = top_idx.to(s_dev, non_blocking=True)
            logp_s = logp_t_top.to(s_dev, non_blocking=True)
            pt_s = p_t_top.to(s_dev, non_blocking=True)
            ptail_s = p_t_tail.to(s_dev, non_blocking=True)

            s_out = student(
                input_ids=ids_s,
                attention_mask=msk_s if args.mode == "sft" else None,
                use_cache=False,
                output_router_logits=(need_router_t or cfg.aux_loss_weight > 0),
                output_hidden_states=need_hidden,
            )
            kl_l, ce_l = compute_fused_logit_kl_ce(
                s_out.logits,
                lbl_s,
                top_idx_s,
                logp_s,
                pt_s,
                ptail_s,
                T=cfg.kl_temperature,
                chunk_size=cfg.chunk_size,
                use_checkpoint=args.grad_ckpt,
                jsd_beta=cfg.jsd_beta,
            )
            tok_mask = (lbl_s != -100) if args.mode == "sft" else None
            rkl_l = (
                compute_router_kl_loss(s_out.router_logits, t_router_targets, token_mask=tok_mask)
                if (need_router_t and s_out.router_logits is not None)
                else torch.zeros((), device=s_dev)
            )
            hid_l = (
                compute_hidden_cosine_loss(
                    s_out.hidden_states, t_hidden_targets, hidden_layers, token_mask=tok_mask
                )
                if (need_hidden and s_out.hidden_states is not None)
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
                + cfg.hidden_align_weight * hid_l
                + cfg.aux_loss_weight * aux_l
            )
            (loss / args.grad_accum).backward()

            acc_loss += float(loss.item()) / args.grad_accum
            acc_kl += float(kl_l.item()) / args.grad_accum
            acc_ce += float(ce_l.item()) / args.grad_accum
            acc_rkl += float(rkl_l.item()) / args.grad_accum
            acc_hid += float(hid_l.item()) / args.grad_accum
            acc_aux += float(aux_l.item()) / args.grad_accum
            del s_out, loss, kl_l, ce_l, rkl_l, hid_l, aux_l

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
                f"(kl={acc_kl:.4f} ce={acc_ce:.4f} hid={acc_hid:.4f} rkl={acc_rkl:.4f} aux={acc_aux:.4f}) | "
                f"gnorm={gnorm:.2f} lr={opt.param_groups[0]['lr']:.2e} | "
                f"{dt:.1f}s ({toks_step/max(dt,1e-3):.0f} tok/s) | vram S={vram_s:.1f}G T={vram_t:.1f}G",
                flush=True,
            )

        if step % args.save_steps == 0 or step == args.max_steps:
            ck_dir = os.path.join(args.out, f"checkpoint-{step}")
            save_merged_native_moe(student, ck_dir, tokenizer=tokenizer)
            if os.path.isfile(cmap_path):
                import shutil
                shutil.copy2(cmap_path, os.path.join(ck_dir, "lrc_cluster_map.json"))
            print(f"[recover] saved merged 100% native HF MoE checkpoint -> {ck_dir}", flush=True)

    print(f"[recover] DONE in {(time.time()-t_start)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
