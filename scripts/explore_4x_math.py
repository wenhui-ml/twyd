#!/usr/bin/env python3
"""Pure-math, training-free (zero backprop) exploration for 4x OLMoE compression (64 -> 16, top-8).

Tests 5 principled mathematical & feature-space mechanisms on OLMoE-1B-7B (2.09B native output):
  Arm 0: Baseline Direct LRC v2 (prose-only calib, 3-sweep ridge, tau=1.0, gamma=0, no transplant)
  Arm 1: + Multi-Domain Balanced Calibration (50% FineWeb+Cosmopedia + 50% open SFT/QA)
  Arm 2: + Per-Layer Router Temperature tau_l (absorbed into gate.weight) + 10-sweep Converged Ridge
  Arm 3: + Cross-Layer Residual Trajectory Error Feedback (gamma = 0.5)
  Arm 4: + Cross-Expert SwiGLU Neuron Transplantation (m_swap = 256 / 384 nonlinear keys from 48 pruned experts)
  Arm 5: + Pure-Math 2-Step Progressive Halving (64 -> 32 [closed-form] -> 16 [closed-form], zero training)

Saves the winning 100% native OlmoeForCausalLM checkpoint for full PRESET=paper evaluation.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compress_moe_lrc import eval_heldout_ce, load_calib_batches
from twyd.moe_compress import OlmoeAdapter, profile_moe_layer, select_top_k_experts
from twyd.moe_compress.lrc_solver import solve_lrc_down_proj
from twyd.moe_compress.pipeline import _advance_hidden_states, _layer_pre_mlp_inputs, _run_layer_forward_sequential


def load_sft_calib_batches(
    tokenizer,
    sft_file: str,
    num_seqs: int = 32,
    seq_len: int = 512,
    batch_size: int = 2,
    skip_rows: int = 0,
) -> List[torch.Tensor]:
    """Load open instruction/QA text sequences from sft_paper8.jsonl (0% eval overlap)."""
    eos = tokenizer.eos_token_id
    buf: List[int] = []
    seqs: List[torch.Tensor] = []
    with open(sft_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            clean = [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in msgs
                if m.get("role") in ("system", "user", "assistant") and str(m.get("content", "")).strip()
            ]
            if not clean:
                continue
            if getattr(tokenizer, "chat_template", None):
                txt = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
            else:
                txt = "\n".join(f"{m['role']}: {m['content']}" for m in clean)
            buf.extend(tokenizer.encode(txt, add_special_tokens=False))
            if eos is not None:
                buf.append(int(eos))
            while len(buf) >= seq_len and len(seqs) < num_seqs:
                seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
            if len(seqs) >= num_seqs:
                break
    return [torch.stack(seqs[i : i + batch_size], dim=0) for i in range(0, len(seqs), batch_size)]


@torch.no_grad()
def eval_batches_ce_and_kl(
    model: nn.Module,
    batches: Sequence[torch.Tensor],
    device: str,
    ref_logits: Optional[List[torch.Tensor]] = None,
) -> Tuple[float, float, float, List[torch.Tensor]]:
    tot_nll = 0.0
    tot_kl = 0.0
    tot_tok = 0
    outs: List[torch.Tensor] = []
    for idx, b in enumerate(batches):
        ids = b.to(device)
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        tot_nll += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum").item())
        tot_tok += int(labels.numel())
        if ref_logits is None:
            outs.append(logits.to(torch.bfloat16).cpu())
        else:
            ref = ref_logits[idx].to(device=device, dtype=torch.float32)
            pt = F.softmax(ref, dim=-1)
            kl = (pt * (F.log_softmax(ref, dim=-1) - F.log_softmax(logits, dim=-1))).sum(dim=-1).sum()
            tot_kl += float(kl.item())
    ce = tot_nll / max(tot_tok, 1)
    return ce, math.exp(min(ce, 50.0)), (tot_kl / max(tot_tok, 1) if ref_logits is not None else 0.0), outs


@torch.no_grad()
def _select_best_router_tau(
    adapter: OlmoeAdapter,
    x_flat: torch.Tensor,
    y_target: torch.Tensor,
    router_slice: torch.Tensor,
    gate_up: torch.Tensor,
    down_init: torch.Tensor,
    act_fn: nn.Module,
    top_k: int,
    norm_topk: bool,
    taus: Sequence[float],
    val_frac: float,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """Grid-search per-layer router temperature tau (absorbed into gate.weight) via beta-rescaled valMSE."""
    if len(taus) <= 1:
        t0 = float(taus[0]) if taus else 1.0
        return t0, router_slice * t0

    best_tau = 1.0
    best_mse = float("inf")
    for tau in taus:
        r_cand = router_slice * float(tau)
        # Fast evaluation using beta-rescale only (0 ridge sweeps)
        res = solve_lrc_down_proj(
            adapter=adapter,
            x_flat=x_flat,
            y_target=y_target,
            router_weight=r_cand,
            gate_up_proj=gate_up,
            down_proj_init=down_init,
            act_fn=act_fn,
            top_k=top_k,
            norm_topk_prob=norm_topk,
            val_frac=val_frac,
            alphas=(),
            n_sweeps=0,
            device=device,
        )
        if res.final_val_rel_mse < best_mse:
            best_mse = res.final_val_rel_mse
            best_tau = float(tau)
    return best_tau, router_slice * best_tau


@torch.no_grad()
def transplant_swiglu_neurons(
    adapter: OlmoeAdapter,
    x_flat: torch.Tensor,
    y_target: torch.Tensor,
    router_w: torch.Tensor,
    all_gate_up: torch.Tensor,   # [E, 2*d_ff, H]
    all_down: torch.Tensor,      # [E, H, d_ff]
    act_fn: nn.Module,
    survivors: List[int],
    pruned: List[int],
    top_k: int,
    norm_topk: bool,
    m_swap: int,
    val_frac: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-expert nonlinear SwiGLU neuron transplantation (100% closed-form, native shapes).

    For each surviving expert j:
      1. Identify the `m_swap` lowest-energy SwiGLU neurons inside host j on j's active fit tokens.
      2. Compute the current layer residual R_j on j's active fit tokens (after initial beta-rescale).
      3. Score ALL donor neurons from the pruned experts (or top correlated donor pool) by their
         exact 1D residual-variance reduction `||h_u^T R_j||^2 / (||h_u||^2 + eps)`.
      4. Transplant the top `m_swap` non-redundant donor `(w_gate, w_up)` pairs into host j's
         weakest `m_swap` slots, initializing their `w_down` column to exact 1D projection.
    """
    K = len(survivors)
    E, two_dff, H = all_gate_up.shape
    d_ff = all_down.shape[2]
    orig_dtype = all_gate_up.dtype
    N = x_flat.shape[0]
    n_val = max(1, int(round(N * val_frac))) if N >= 8 else 0
    n_fit = N - n_val

    surv_idx = torch.tensor(survivors, dtype=torch.long, device=all_gate_up.device)
    new_gu = all_gate_up.index_select(0, surv_idx).detach().to(device=device, dtype=orig_dtype).clone()
    new_dn_init = all_down.index_select(0, surv_idx).detach().to(device=device, dtype=orig_dtype).clone()

    if m_swap <= 0 or not pruned:
        return new_gu, new_dn_init

    # First run fast beta-rescale so residual R is true orthogonal missing output (not scale error)
    beta_res = solve_lrc_down_proj(
        adapter=adapter,
        x_flat=x_flat,
        y_target=y_target,
        router_weight=router_w,
        gate_up_proj=new_gu,
        down_proj_init=new_dn_init,
        act_fn=act_fn,
        top_k=top_k,
        norm_topk_prob=norm_topk,
        val_frac=val_frac,
        alphas=(),
        n_sweeps=0,
        device=device,
    )
    new_dn = beta_res.down_proj.to(device=device, dtype=orig_dtype).clone()

    x_fit = x_flat[:n_fit].to(device=device, dtype=orig_dtype)
    y_fit = y_target[:n_fit].to(device=device, dtype=torch.float32)
    _, _, tw_fit, ti_fit = adapter.route_with_weight(x_fit, router_w.to(device), top_k, norm_topk)

    # Current beta-rescaled layer output on fit tokens
    y_cur = torch.zeros_like(y_fit)
    for j in range(K):
        tp, sp = torch.where(ti_fit == j)
        if tp.numel() == 0:
            continue
        h = adapter.expert_hidden(x_fit[tp], new_gu[j], act_fn)
        y_j = F.linear(h, new_dn[j]).to(torch.float32) * tw_fit[tp, sp].unsqueeze(-1).to(torch.float32)
        y_cur.index_add_(0, tp, y_j)

    resid_all = y_fit - y_cur  # [n_fit, H]

    # Pre-fetch donor weights on device
    pruned_idx = torch.tensor(pruned, dtype=torch.long, device=all_gate_up.device)
    donor_gu = all_gate_up.index_select(0, pruned_idx).to(device=device, dtype=orig_dtype)  # [P, 2*d_ff, H]
    P_exp = donor_gu.shape[0]

    # Split donor gate and up for clean neuron indexing: [P*d_ff, H]
    donor_gate = donor_gu[:, :d_ff, :].reshape(P_exp * d_ff, H)
    donor_up = donor_gu[:, d_ff:, :].reshape(P_exp * d_ff, H)
    n_donors = donor_gate.shape[0]

    # Track already-transplanted donors across the 16 hosts so experts stay diverse!
    used_donor_mask = torch.zeros(n_donors, dtype=torch.bool, device=device)

    for j in range(K):
        tp, sp = torch.where(ti_fit == j)
        if tp.numel() < 32:
            continue
        # Subsample up to 2048 active tokens of host j for fast exact scoring
        if tp.numel() > 2048:
            stride = tp.numel() // 2048
            tp = tp[::stride][:2048]
            sp = sp[::stride][:2048]
        xj = x_fit[tp]                                  # [M, H]
        wj = tw_fit[tp, sp].to(torch.float32).unsqueeze(-1)  # [M, 1]
        rj = resid_all[tp]                              # [M, H]

        # 1. Rank host j's own 1024 neurons by actual weighted output energy on xj
        hj = adapter.expert_hidden(xj, new_gu[j], act_fn).to(torch.float32) * wj  # [M, d_ff]
        dn_col_sq = (new_dn[j].float() ** 2).sum(dim=0)  # [d_ff]
        host_energy = (hj * hj).sum(dim=0) * dn_col_sq
        weak_slots = torch.argsort(host_energy)[:m_swap]  # lowest-energy slots to replace

        # 2. Score all donor neurons in chunks by exact residual explanation ||h_u^T rj||^2 / (||h_u||^2 + eps)
        donor_scores = torch.full((n_donors,), -1.0, dtype=torch.float32, device=device)
        chunk_d = 4096
        for d0 in range(0, n_donors, chunk_d):
            d1 = min(d0 + chunk_d, n_donors)
            g_c = F.linear(xj, donor_gate[d0:d1])
            u_c = F.linear(xj, donor_up[d0:d1])
            h_c = (act_fn(g_c) * u_c).to(torch.float32) * wj  # [M, C]
            h_norm_sq = (h_c * h_c).sum(dim=0)                # [C]
            proj = h_c.T @ rj                                  # [C, H]
            score_c = (proj * proj).sum(dim=-1) / (h_norm_sq + 1e-4)
            donor_scores[d0:d1] = score_c

        donor_scores.masked_fill_(used_donor_mask, -1.0)
        best_donors = torch.topk(donor_scores, k=min(m_swap, n_donors)).indices
        used_donor_mask[best_donors] = True

        # 3. Transplant exact (w_gate, w_up) pairs into weak slots & init w_down by 1D projection
        sel_g = donor_gate[best_donors]
        sel_u = donor_up[best_donors]
        new_gu[j, weak_slots, :] = sel_g
        new_gu[j, d_ff + weak_slots, :] = sel_u

        # Exact 1D closed-form init for transplanted down_proj columns
        h_new = (act_fn(F.linear(xj, sel_g)) * F.linear(xj, sel_u)).to(torch.float32) * wj  # [M, m_swap]
        denom = (h_new * h_new).sum(dim=0).clamp_min(1e-4)  # [m_swap]
        w_dn_new = (rj.T @ h_new) / denom.unsqueeze(0)      # [H, m_swap]
        # Damped init (0.5x) before full coupled Gauss-Seidel Ridge refines all 1024 columns jointly
        new_dn[j, :, weak_slots] = (0.5 * w_dn_new).to(orig_dtype)

    return new_gu, new_dn


@torch.no_grad()
def run_math_arm(
    base_model: nn.Module,
    calib_batches: Sequence[torch.Tensor],
    holdout_prose: Sequence[torch.Tensor],
    holdout_qa: Sequence[torch.Tensor],
    ref_prose_logits: List[torch.Tensor],
    ref_qa_logits: List[torch.Tensor],
    target_experts: int | Sequence[int] = 16,
    target_top_k: int = 8,
    taus: Sequence[float] = (1.0,),
    ridge_sweeps: int = 3,
    ridge_alphas: Sequence[float] = (0.2, 1.0, 5.0, 20.0, 80.0),
    gamma_res: float = 0.0,
    m_swap: int = 0,
    device_str: str = "cuda:0",
    save_dir: Optional[str] = None,
    tokenizer=None,
    tag: str = "arm",
) -> Dict:
    """Run one pure-math training-free compression arm and evaluate both Prose & QA holdouts."""
    t0 = time.time()
    dev = torch.device(device_str)
    model = copy.deepcopy(base_model).to(dev).eval()
    adapter = OlmoeAdapter()
    moe_layers = list(adapter.iter_moe_layers(model))

    # Maintain BOTH teacher hidden states (for gamma_res trajectory error feedback) and student hidden states
    h_stu, pe_list, pid_list = _run_layer_forward_sequential(model, calib_batches, dev)
    h_tea = [t.clone() for t in h_stu] if gamma_res > 0 else None

    val_mses = []
    init_mses = []
    taus_picked = []
    cmaps = {}

    for layer_idx, layer_mod, moe_block in moe_layers:
        orig_e = adapter.get_num_experts(moe_block)
        orig_k = adapter.get_top_k(moe_block)
        norm_topk = adapter.get_norm_topk_prob(moe_block)
        router_w = adapter.get_router_weight(moe_block)
        gu, dn, act_fn = adapter.get_expert_weights(moe_block)

        # Student pre-MLP input
        x_stu = _layer_pre_mlp_inputs(layer_mod, h_stu, pe_list, pid_list, dev)

        # Profile teacher MoE on x_stu
        prof = profile_moe_layer(adapter, moe_block, x_stu, layer_idx=layer_idx, chunk_size=512, probe_tokens=512, device=dev)
        y_target = prof.teacher_y

        # Optional residual trajectory error feedback: match Teacher's post-layer state
        if gamma_res > 0 and h_tea is not None:
            # Advance teacher copy of THIS layer before compressing it
            h_tea_next = _advance_hidden_states(layer_mod, h_tea, pe_list, pid_list, dev)
            # Compute student post-attn residual (before MLP) = layer_out - teacher_y_on_stu
            # Exact post-attn residual on h_stu:
            h_stu_with_tmoe = _advance_hidden_states(layer_mod, h_stu, pe_list, pid_list, dev)
            flat_t_next = torch.cat([t.reshape(-1, t.shape[-1]) for t in h_tea_next], dim=0).float()
            flat_s_with_t = torch.cat([t.reshape(-1, t.shape[-1]) for t in h_stu_with_tmoe], dim=0).float()
            # Trajectory correction delta = h_tea_next - h_stu_with_tmoe
            y_target = (prof.teacher_y.float() + float(gamma_res) * (flat_t_next - flat_s_with_t)).to(prof.teacher_y.dtype)
            h_tea = h_tea_next

        k_layer = int(target_experts[layer_idx]) if isinstance(target_experts, (list, tuple)) else int(target_experts)
        top_k_layer = min(int(target_top_k), k_layer)
        sel = select_top_k_experts(prof, target_experts=k_layer, method="lrc_cluster")
        surv_idx = torch.tensor(sel.survivors, dtype=torch.long, device=router_w.device)
        r_slice = router_w.index_select(0, surv_idx).detach().clone()
        gu_init = gu.index_select(0, surv_idx).detach().clone()
        dn_init = dn.index_select(0, surv_idx).detach().clone()

        # 1. Per-layer router temperature search
        best_tau, r_new = _select_best_router_tau(
            adapter, x_stu, y_target, r_slice, gu_init, dn_init, act_fn,
            top_k_layer, norm_topk, taus, 0.15, dev,
        )
        taus_picked.append(best_tau)

        # 2. Optional Cross-Expert SwiGLU Neuron Transplantation
        if m_swap > 0:
            gu_new, dn_pre = transplant_swiglu_neurons(
                adapter, x_stu, y_target, r_new, gu, dn, act_fn,
                sel.survivors, sel.pruned, top_k_layer, norm_topk, m_swap, 0.15, dev,
            )
        else:
            gu_new, dn_pre = gu_init, dn_init

        # 3. Coupled Gauss-Seidel LRC Ridge Solve
        s_res = solve_lrc_down_proj(
            adapter=adapter,
            x_flat=x_stu,
            y_target=y_target,
            router_weight=r_new,
            gate_up_proj=gu_new,
            down_proj_init=dn_pre,
            act_fn=act_fn,
            top_k=top_k_layer,
            norm_topk_prob=norm_topk,
            val_frac=0.15,
            alphas=ridge_alphas,
            n_sweeps=ridge_sweeps,
            damping_step=0.85,
            device=dev,
        )

        adapter.rebuild_moe_block_(
            model, layer_idx, r_new, gu_new, s_res.down_proj, k_layer, top_k_layer
        )
        h_stu = _advance_hidden_states(layer_mod, h_stu, pe_list, pid_list, dev)

        init_mses.append(s_res.init_val_rel_mse)
        val_mses.append(s_res.final_val_rel_mse)
        cmaps[str(layer_idx)] = {"survivors": sel.survivors, "cluster_map": sel.cluster_map}

    if isinstance(target_experts, int):
        adapter.finalize_config_(model, target_experts, target_top_k)

    ce_p, ppl_p, kl_p, _ = eval_batches_ce_and_kl(model, holdout_prose, device_str, ref_prose_logits)
    ce_q, ppl_q, kl_q, _ = eval_batches_ce_and_kl(model, holdout_qa, device_str, ref_qa_logits)
    elapsed = time.time() - t0

    res = {
        "tag": tag,
        "target_experts": target_experts,
        "target_top_k": target_top_k,
        "mean_init_val_mse": sum(init_mses) / len(init_mses),
        "mean_final_val_mse": sum(val_mses) / len(val_mses),
        "prose_ce": ce_p,
        "prose_ppl": ppl_p,
        "prose_kl": kl_p,
        "qa_ce": ce_q,
        "qa_ppl": ppl_q,
        "qa_kl": kl_q,
        "mean_tau": sum(taus_picked) / len(taus_picked),
        "elapsed_sec": elapsed,
    }
    print(
        f"[RESULT] {tag:34s} | valMSE={res['mean_final_val_mse']:.4f} | "
        f"Prose PPL={ppl_p:6.2f} KL={kl_p:.4f} | QA PPL={ppl_q:6.2f} KL={kl_q:.4f} | "
        f"tau={res['mean_tau']:.2f} | {elapsed:.0f}s",
        flush=True,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        json.dump(
            {"orig_experts": 64, "new_experts": target_experts, "orig_top_k": 8, "new_top_k": target_top_k, "layers": cmaps},
            open(os.path.join(save_dir, "lrc_cluster_map.json"), "w"),
            indent=2,
        )
        json.dump(res, open(os.path.join(save_dir, "math_report.json"), "w"), indent=2)

    del model
    torch.cuda.empty_cache()
    return res


def main() -> None:
    device_str = "cuda:0"
    teacher_path = os.path.expanduser("~/models/OLMoE-1B-7B-0125-Instruct")
    tok = AutoTokenizer.from_pretrained(teacher_path)
    print(f"[explore] loading Teacher {teacher_path} to CPU base ...", flush=True)
    base_cpu = AutoModelForCausalLM.from_pretrained(
        teacher_path, dtype=torch.bfloat16, experts_implementation="eager", low_cpu_mem_usage=True, device_map={"": "cpu"}
    ).eval()

    # Calibration sets: (A) prose-only 64x512, (B) balanced 50% prose + 50% open SFT/QA (64x512)
    prose_calib_64 = load_calib_batches(
        tok, [os.path.join(ROOT, "data/fw_score4_200M.jsonl"), os.path.join(ROOT, "data/pt_cosmopedia_v3_pool.jsonl")],
        num_seqs=64, seq_len=512, batch_size=2,
    )
    prose_calib_32 = prose_calib_64[:16]
    sft_calib_32 = load_sft_calib_batches(tok, os.path.join(ROOT, "data/sft_paper8.jsonl"), num_seqs=32, seq_len=512, batch_size=2, skip_rows=0)
    # Interleave prose + SFT so fit (85%) and val (15%) splits BOTH contain 50/50 prose + QA!
    bal_calib_64 = []
    for a, b in zip(prose_calib_32, sft_calib_32):
        bal_calib_64.append(a)
        bal_calib_64.append(b)

    # Disjoint holdouts: FineWeb-Edu holdout (prose) + disjoint tail of sft_paper8 (open QA/Instruct holdout)
    holdout_prose = load_calib_batches(tok, [os.path.join(ROOT, "data/fineweb_edu_holdout.jsonl")], num_seqs=32, seq_len=512, batch_size=2)
    holdout_qa = load_sft_calib_batches(tok, os.path.join(ROOT, "data/sft_paper8.jsonl"), num_seqs=24, seq_len=512, batch_size=2, skip_rows=50000)

    # Teacher reference logits
    t_gpu = copy.deepcopy(base_cpu).to(device_str).eval()
    t_ce_p, t_ppl_p, _, ref_prose = eval_batches_ce_and_kl(t_gpu, holdout_prose, device_str)
    t_ce_q, t_ppl_q, _, ref_qa = eval_batches_ce_and_kl(t_gpu, holdout_qa, device_str)
    print(f"[TEACHER] E=64 k=8 | Prose CE={t_ce_p:.4f} PPL={t_ppl_p:.2f} | QA CE={t_ce_q:.4f} PPL={t_ppl_q:.2f}\n", flush=True)
    del t_gpu
    torch.cuda.empty_cache()

    # Reuse measured Arm0/1/2 baselines as reference, now run Arm3..Arm6 on the winning tau=1.0, 3-sweep base
    results = [
        {"tag": "Arm0_ProseOnly_Direct16_k8", "mean_final_val_mse": 0.2397, "prose_ppl": 51.78, "prose_kl": 2.1682, "qa_ppl": 39.14, "qa_kl": 2.5596},
        {"tag": "Arm1_BalancedCalib_Direct16_k8", "mean_final_val_mse": 0.2181, "prose_ppl": 53.88, "prose_kl": 2.2017, "qa_ppl": 38.46, "qa_kl": 2.3015},
        {"tag": "Arm2_Bal_Tau_8SweepRidge", "mean_final_val_mse": 0.2396, "prose_ppl": 55.00, "prose_kl": 2.2333, "qa_ppl": 40.37, "qa_kl": 2.3452},
    ]

    # Arm 3: + Cross-Layer Residual Trajectory Error Feedback (gamma = 0.5)
    results.append(run_math_arm(
        base_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=16, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.5, m_swap=0,
        tag="Arm3_Bal_3Sweep_ResGamma0.5",
    ))

    # Arm 4a: + Cross-Expert SwiGLU Neuron Transplantation (m_swap = 128 / 1024)
    results.append(run_math_arm(
        base_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=16, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.0, m_swap=128,
        tag="Arm4a_Bal_Transplant128",
    ))

    # Arm 4b: + Cross-Expert SwiGLU Neuron Transplantation (m_swap = 256 / 1024)
    results.append(run_math_arm(
        base_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=16, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.0, m_swap=256,
        tag="Arm4b_Bal_Transplant256",
    ))

    # Arm 5: Pure-Math 2-Step Progressive Halving (64->32 closed-form -> 16 closed-form, ZERO training!)
    tmp32 = os.path.join(ROOT, "output/moe_lrc/_tmp_math_e32")
    run_math_arm(
        base_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=32, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.0, m_swap=0, save_dir=tmp32, tokenizer=tok,
        tag="Arm5_Step1_64to32_MathOnly",
    )
    mid32_cpu = AutoModelForCausalLM.from_pretrained(
        tmp32, dtype=torch.bfloat16, experts_implementation="eager", low_cpu_mem_usage=True, device_map={"": "cpu"}
    ).eval()
    results.append(run_math_arm(
        mid32_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=16, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.0, m_swap=0,
        tag="Arm5_Step2_32to16_PureMathProg",
    ))
    import shutil
    shutil.rmtree(tmp32, ignore_errors=True)

    # Arm 6: Feature/eRank-Driven Non-Uniform Per-Layer Allocation (EXACT SAME 256 total experts = 2.09B!)
    # Low-rank L01..L04,L08,L09 donate experts to high-rank L00,L05,L06,L07,L11..L15:
    # sum = 20+8+4+8+10+22+20+24+12+10+16+20+24+20+22+16 = 256 (exact 16.0 avg = 4x!)
    nonuniform_k = [20, 8, 4, 8, 10, 22, 20, 24, 12, 10, 16, 20, 24, 20, 22, 16]
    assert sum(nonuniform_k) == 256
    results.append(run_math_arm(
        base_cpu, bal_calib_64, holdout_prose, holdout_qa, ref_prose, ref_qa,
        target_experts=nonuniform_k, target_top_k=8, taus=(1.0,), ridge_sweeps=3,
        gamma_res=0.0, m_swap=0,
        tag="Arm6_eRankNonUniform256_Same2.09B",
    ))

    print("\n=== SUMMARY OF ALL PURE-MATH 4x (64->16, top-8, ZERO TRAINING) ARMS ===")
    results_sorted = sorted(results, key=lambda r: r["prose_kl"] + r["qa_kl"])
    for r in results_sorted:
        print(
            f"{r['tag']:34s} | valMSE={r['mean_final_val_mse']:.4f} | "
            f"Prose PPL={r['prose_ppl']:6.2f} KL={r['prose_kl']:.4f} | "
            f"QA PPL={r['qa_ppl']:6.2f} KL={r['qa_kl']:.4f} | SumKL={r['prose_kl']+r['qa_kl']:.4f}"
        )

    out_json = os.path.join(ROOT, "output/moe_lrc/math_4x_exploration.json")
    json.dump({"teacher": {"prose_ppl": t_ppl_p, "qa_ppl": t_ppl_q}, "arms": results}, open(out_json, "w"), indent=2)
    print(f"\nSaved full report -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
