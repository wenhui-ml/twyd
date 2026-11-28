#!/usr/bin/env python3
"""Deep mechanistic & forensic analysis of OLMoE-1B-7B expert compression gaps.

Measures 5 concrete diagnostics directly on GPU (read-only analysis):
  1. Per-layer expert redundancy spectrum (effective rank, top-16/32 energy,
     pairwise functional cosine similarity across all 16 layers).
  2. Per-neuron energy concentration inside each expert (1024 SwiGLU neurons):
     tests whether pruned experts hold high-value nonlinear keys while surviving
     experts hold many low-energy dead neurons (Cluster Neuron-Packing potential).
  3. Per-layer single-layer replacement damage (hybrid Teacher with layer l
     swapped from 2x-Rec20M or 4x-KDSFT): pinpoints WHICH layers cause the CE
     and QA/MMLU degradation.
  4. Domain routing shift: compares Teacher routing/energy on prose calibration
     (FineWeb+Cosmopedia) vs MMLU/CSQA prompts (diagnostic only!) to quantify
     how much QA-specific expert mass was killed by prose-only selection.
  5. Logit margin & confidence breakdown on MMLU/CSQA errors (Teacher vs 2x vs 4x).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compress_moe_lrc import eval_heldout_ce, load_calib_batches
from twyd.moe_compress import OlmoeAdapter, profile_moe_layer


def _load_olmoe(path: str, device: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        experts_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=False,
    ).eval()


def _load_qa_prompt_batches(tokenizer, sample_files: List[str], num_seqs: int = 32, seq_len: int = 256):
    """Load diagnostic QA prompt text from existing lm_eval sample logs (analysis only)."""
    texts = []
    for path in sample_files:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                o = json.loads(line)
                args = o.get("arguments")
                if isinstance(args, dict) and "gen_args_0" in args:
                    texts.append(str(args["gen_args_0"].get("arg_0", "")))
                elif isinstance(args, list) and args:
                    first = args[0]
                    if isinstance(first, list) and first:
                        texts.append(str(first[0]))
                    elif isinstance(first, dict) and "gen_args_0" in first:
                        texts.append(str(first["gen_args_0"].get("arg_0", "")))
                if len(texts) >= num_seqs * 8:
                    break
    buf = []
    seqs = []
    eos = tokenizer.eos_token_id
    for t in texts:
        if not t:
            continue
        buf.extend(tokenizer.encode(t, add_special_tokens=False))
        if eos is not None:
            buf.append(int(eos))
        while len(buf) >= seq_len and len(seqs) < num_seqs:
            seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
        if len(seqs) >= num_seqs:
            break
    return [torch.stack(seqs[i : i + 2], dim=0) for i in range(0, len(seqs), 2)]


@torch.no_grad()
def main() -> None:
    device = "cuda:0"
    teacher_path = os.path.expanduser("~/models/OLMoE-1B-7B-0125-Instruct")
    rec32_path = os.path.join(ROOT, "output/moe_lrc/olmoe-lrc-v2-e32-rec20M/checkpoint-600")
    kdsft16_path = os.path.join(ROOT, "output/moe_lrc/olmoe-prog-e16-k4-pt50M-kdsft/checkpoint-350")
    cmap32 = json.load(open(os.path.join(ROOT, "output/moe_lrc/olmoe-lrc-v2-e32/lrc_cluster_map.json")))
    cmap16 = json.load(open(os.path.join(ROOT, "output/moe_lrc/olmoe-prog-e16-k4/lrc_cluster_map_64to16.json")))

    tok = AutoTokenizer.from_pretrained(teacher_path)
    teacher = _load_olmoe(teacher_path, device)
    adapter = OlmoeAdapter()

    # Load prose calibration (32 seqs x 512 = 16,384 tok) and diagnostic QA prompts
    prose_batches = load_calib_batches(
        tok,
        [os.path.join(ROOT, "data/fw_score4_200M.jsonl"), os.path.join(ROOT, "data/pt_cosmopedia_v3_pool.jsonl")],
        num_seqs=32,
        seq_len=512,
        batch_size=2,
    )
    import glob
    qa_files = sorted(glob.glob(os.path.join(ROOT, "output/lm_eval/paper_olmoe_teacher_e64/**/samples_commonsense_qa_*.jsonl"), recursive=True))
    qa_files += sorted(glob.glob(os.path.join(ROOT, "output/lm_eval/paper_olmoe_teacher_e64/**/samples_mmlu_high_school_*.jsonl"), recursive=True))[:6]
    qa_batches = _load_qa_prompt_batches(tok, qa_files, num_seqs=32, seq_len=256)
    print(f"[diag] prose={len(prose_batches)*2*512} tok | qa_diag={len(qa_batches)*2*256} tok", flush=True)

    # Capture per-layer inputs for prose and QA in 2 fast forward passes
    def capture_inputs(batches):
        cap = {i: [] for i in range(len(teacher.model.layers))}
        hooks = []
        for i, layer in enumerate(teacher.model.layers):
            def _mk(idx):
                def _h(_m, args):
                    cap[idx].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())
                return _h
            hooks.append(layer.mlp.register_forward_pre_hook(_mk(i)))
        for b in batches:
            teacher.model(input_ids=b.to(device), use_cache=False)
        for h in hooks:
            h.remove()
        return {i: torch.cat(v, dim=0) for i, v in cap.items()}

    x_prose = capture_inputs(prose_batches)
    x_qa = capture_inputs(qa_batches)

    print("\n=== [DIAG 1 & 2 & 4] PER-LAYER EXPERT SPECTRUM, NEURON SPARSITY & PROSE-VS-QA ROUTING SHIFT ===")
    print(
        f"{'L':>2s} | {'eRank':>5s} | {'sim>0.5':>7s} | {'top16E':>6s} | {'top32E':>6s} | "
        f"{'neu90%':>6s} | {'neu50%':>6s} | {'QA_ret32':>8s} | {'QA_ret16':>8s} | {'Prs_ret32':>9s} | {'rankCorr':>8s}"
    )
    print("-" * 105)

    layer_report = []
    for i, layer in enumerate(teacher.model.layers):
        moe = layer.mlp
        prof_p = profile_moe_layer(adapter, moe, x_prose[i], layer_idx=i, chunk_size=512, probe_tokens=256, device=device)
        prof_q = profile_moe_layer(adapter, moe, x_qa[i], layer_idx=i, chunk_size=512, probe_tokens=256, device=device)

        # 1. Effective rank of 64x64 Gram matrix G (how many independent expert directions exist)
        evals = torch.linalg.eigvalsh(prof_p.gram.float()).clamp_min(0.0)
        p_ev = evals / evals.sum().clamp_min(1e-12)
        erank = float(torch.exp(-(p_ev * p_ev.clamp_min(1e-12).log()).sum()).item())

        # Pairwise functional similarity off-diagonal > 0.5
        sim = prof_p.func_sim.clone()
        sim.fill_diagonal_(-1.0)
        high_sim_pairs = int((sim > 0.5).sum().item()) // 2
        max_sim = float(sim.max().item())

        e_sorted = torch.sort(prof_p.output_energy, descending=True).values
        e_tot = float(e_sorted.sum().item())
        top16_e = float(e_sorted[:16].sum().item()) / e_tot
        top32_e = float(e_sorted[:32].sum().item()) / e_tot

        # 2. Per-neuron energy concentration inside each expert (1024 neurons)
        # Measure on 1024 probe tokens across first 16 experts for speed
        gu, dn, act_fn = adapter.get_expert_weights(moe)
        xp_sub = x_prose[i][:: max(1, x_prose[i].shape[0] // 1024)][:1024].to(device=device, dtype=gu.dtype)
        dn_col_sq = (dn.float() ** 2).sum(dim=1)  # [E, d_ff]: ||w_down[:, u]||^2
        neu90_list, neu50_list = [], []
        for e in range(0, 64, 4):
            h = adapter.expert_hidden(xp_sub, gu[e], act_fn).float()  # [P, d_ff]
            # Exact per-neuron expected squared output contribution: E[h_u^2] * ||w_down[:, u]||^2
            neu_e = (h * h).mean(dim=0) * dn_col_sq[e]
            ns = torch.sort(neu_e, descending=True).values
            csum = torch.cumsum(ns, dim=0) / ns.sum().clamp_min(1e-12)
            neu50_list.append(int((csum < 0.50).sum().item()) + 1)
            neu90_list.append(int((csum < 0.90).sum().item()) + 1)
        mean_neu90 = sum(neu90_list) / len(neu90_list)
        mean_neu50 = sum(neu50_list) / len(neu50_list)

        # 4. Domain shift: how much of QA energy & Prose energy is retained by our kept 32 & 16 experts?
        s32 = torch.tensor(cmap32["layers"][str(i)]["survivors"], dtype=torch.long)
        s16 = torch.tensor(cmap16["layers"][str(i)]["survivors"], dtype=torch.long)
        qa_ret32 = float(prof_q.output_energy[s32].sum() / prof_q.output_energy.sum().clamp_min(1e-12))
        qa_ret16 = float(prof_q.output_energy[s16].sum() / prof_q.output_energy.sum().clamp_min(1e-12))
        prs_ret32 = float(prof_p.output_energy[s32].sum() / prof_p.output_energy.sum().clamp_min(1e-12))

        # Spearman-like rank correlation of expert gate_mass between Prose and QA
        rp = torch.argsort(torch.argsort(prof_p.gate_mass)).float()
        rq = torch.argsort(torch.argsort(prof_q.gate_mass)).float()
        rp_c = rp - rp.mean()
        rq_c = rq - rq.mean()
        rcorr = float(((rp_c * rq_c).sum() / (rp_c.norm() * rq_c.norm()).clamp_min(1e-12)).item())

        print(
            f"{i:2d} | {erank:5.1f} | {high_sim_pairs:3d}({max_sim:.2f}) | {top16_e:6.3f} | {top32_e:6.3f} | "
            f"{mean_neu90:5.0f}/1k | {mean_neu50:5.0f}/1k | {qa_ret32:8.3f} | {qa_ret16:8.3f} | {prs_ret32:9.3f} | {rcorr:8.3f}",
            flush=True,
        )
        layer_report.append({
            "layer": i, "erank": erank, "high_sim_pairs": high_sim_pairs, "max_sim": max_sim,
            "top16_e": top16_e, "top32_e": top32_e, "neu90": mean_neu90, "neu50": mean_neu50,
            "qa_ret32": qa_ret32, "qa_ret16": qa_ret16, "prs_ret32": prs_ret32, "rank_corr": rcorr,
        })

    # === [DIAG 3] Single-Layer Swap Damage Profile (Teacher with 1 layer swapped from 2x-Rec20M / 4x-KDSFT) ===
    print("\n=== [DIAG 3] SINGLE-LAYER REPLACEMENT DAMAGE (ONLY LAYER l SWAPPED INTO 64-EXPERT TEACHER) ===")
    holdout_file = os.path.join(ROOT, "data/fineweb_edu_holdout.jsonl")
    holdout_batches = load_calib_batches(tok, [holdout_file], num_seqs=16, seq_len=512, batch_size=2)

    def quick_ce_and_kl(model, batches, ref_logits_list=None):
        tot_nll = 0.0
        tot_kl = 0.0
        tot_tok = 0
        outs = []
        for idx_b, b in enumerate(batches):
            ids = b.to(device)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
            labels = ids[:, 1:]
            tot_nll += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum").item())
            tot_tok += int(labels.numel())
            if ref_logits_list is None:
                outs.append(logits.to(torch.bfloat16).cpu())
            else:
                ref = ref_logits_list[idx_b].to(device=device, dtype=torch.float32)
                pt = F.softmax(ref, dim=-1)
                kl = (pt * (F.log_softmax(ref, dim=-1) - F.log_softmax(logits, dim=-1))).sum(dim=-1).sum()
                tot_kl += float(kl.item())
        return tot_nll / tot_tok, (tot_kl / tot_tok if ref_logits_list is not None else 0.0), outs

    base_ce, _, ref_prose = quick_ce_and_kl(teacher, holdout_batches)
    qa_ce, _, ref_qa = quick_ce_and_kl(teacher, qa_batches[:8])
    print(f"Teacher baseline: holdout_CE={base_ce:.4f} (PPL={math.exp(base_ce):.2f}) | qa_CE={qa_ce:.4f}", flush=True)

    for tag, spath in [("2x-Rec20M (E32)", rec32_path), ("4x-KDSFT (E16,k4)", kdsft16_path)]:
        stu = AutoModelForCausalLM.from_pretrained(
            spath, dtype=torch.bfloat16, experts_implementation="eager", low_cpu_mem_usage=True, device_map={"": "cpu"}
        ).eval()
        print(f"\n--- Single-layer swap from {tag} into Teacher ---")
        print(f"{'L':>2s} | {'dCE_prose':>9s} | {'KL_prose':>8s} | {'dCE_QA':>8s} | {'KL_QA':>8s}")
        for i in range(len(teacher.model.layers)):
            orig_mlp = teacher.model.layers[i].mlp
            stu_mlp = stu.model.layers[i].mlp.to(device)
            teacher.model.layers[i].mlp = stu_mlp
            c_p, k_p, _ = quick_ce_and_kl(teacher, holdout_batches, ref_prose)
            c_q, k_q, _ = quick_ce_and_kl(teacher, qa_batches[:8], ref_qa)
            teacher.model.layers[i].mlp = orig_mlp
            stu_mlp.to("cpu")
            print(f"{i:2d} | {c_p - base_ce:+9.4f} | {k_p:8.4f} | {c_q - qa_ce:+8.4f} | {k_q:8.4f}", flush=True)
        del stu

    # === [DIAG 5] MMLU & CSQA Error Breakdown from Saved Eval Samples ===
    print("\n=== [DIAG 5] MMLU 4-CATEGORY & TASK RETENTION SUMMARY ===")
    for tag, d in [
        ("Teacher-E64", "output/lm_eval/paper_olmoe_teacher_e64"),
        ("2x-Step0-E32", "output/lm_eval/paper_olmoe_lrc_v2_e32"),
        ("2x-Rec20M-E32", "output/lm_eval/paper_olmoe_lrc_v2_e32_rec20M"),
        ("4x-Step0-E16", "output/lm_eval/paper_olmoe_prog_e16_k4"),
        ("4x-PT50M-E16", "output/lm_eval/paper_olmoe_prog_e16_k4_pt50M"),
        ("4x-KDSFT-E16", "output/lm_eval/paper_olmoe_prog_e16_k4_kdsft"),
    ]:
        jf = sorted(glob.glob(os.path.join(ROOT, d, "**", "results_*.json"), recursive=True))[-1]
        r = json.load(open(jf))["results"]
        print(
            f"{tag:15s} | MMLU={r['mmlu']['acc,none']:.4f} "
            f"(STEM={r['mmlu_stem']['acc,none']:.4f} Soc={r['mmlu_social_sciences']['acc,none']:.4f} "
            f"Hum={r['mmlu_humanities']['acc,none']:.4f} Oth={r['mmlu_other']['acc,none']:.4f}) | "
            f"CSQA={r['commonsense_qa']['acc,none']:.4f} BoolQ={r['boolq']['acc,none']:.4f}"
        )


if __name__ == "__main__":
    main()
