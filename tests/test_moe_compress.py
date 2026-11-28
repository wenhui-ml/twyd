"""Unit tests for LRC-MoE Top-K' expert compression (`twyd/moe_compress`)."""
from __future__ import annotations

import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.olmoe import OlmoeConfig, OlmoeForCausalLM

from twyd.moe_compress import (
    MoeCompressConfig,
    OlmoeAdapter,
    compress_moe_model,
    fit_cluster_router,
    profile_moe_layer,
    select_top_k_experts,
    solve_lrc_down_proj,
)


def _tiny_olmoe(
    seed: int = 0,
    num_experts: int = 8,
    top_k: int = 2,
    norm_topk_prob: bool = False,
    num_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 32,
) -> OlmoeForCausalLM:
    torch.manual_seed(seed)
    cfg = OlmoeConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk_prob,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = OlmoeForCausalLM(cfg).eval()
    return model


def _teacher_kl_and_mse(
    teacher: OlmoeForCausalLM, student: OlmoeForCausalLM, batches: list[torch.Tensor]
) -> tuple[float, float]:
    total_kl = 0.0
    total_mse = 0.0
    total_toks = 0
    with torch.no_grad():
        for ids in batches:
            t_logits = teacher(input_ids=ids, use_cache=False).logits.float()
            s_logits = student(input_ids=ids, use_cache=False).logits.float()
            p_t = F.softmax(t_logits, dim=-1)
            log_p_t = F.log_softmax(t_logits, dim=-1)
            log_p_s = F.log_softmax(s_logits, dim=-1)
            total_kl += float((p_t * (log_p_t - log_p_s)).sum(dim=-1).sum().item())
            total_mse += float(((t_logits - s_logits) ** 2).mean(dim=-1).sum().item())
            total_toks += int(ids.numel())
    return total_kl / total_toks, total_mse / total_toks


def test_profiler_gram_and_teacher_y_exact():
    """Exactness of chunked Gram matrix and teacher_y vs naive brute-force forward."""
    model = _tiny_olmoe(seed=7, num_experts=6, top_k=2, norm_topk_prob=False)
    adapter = OlmoeAdapter()
    moe = model.model.layers[0].mlp
    torch.manual_seed(11)
    x = torch.randn(40, 64)

    prof = profile_moe_layer(adapter, moe, x, layer_idx=0, chunk_size=13, probe_tokens=20)

    # Brute-force reference
    with torch.no_grad():
        ref_y = moe(x.unsqueeze(0)).squeeze(0)
        _, _, tw, ti = adapter.route_with_weight(x, moe.gate.weight, 2, False)
        z = torch.zeros(6, 40, 64, dtype=torch.float64)
        for n in range(40):
            for s in range(2):
                e = int(ti[n, s].item())
                ye = adapter.expert_forward(
                    x[n : n + 1], moe.experts.gate_up_proj[e], moe.experts.down_proj[e], moe.experts.act_fn
                )
                z[e, n] = (ye.squeeze(0) * tw[n, s]).double()
        ref_gram = torch.einsum("enh,fnh->ef", z, z)

    assert torch.allclose(prof.teacher_y, ref_y, atol=1e-5, rtol=1e-4)
    assert torch.allclose(prof.gram, ref_gram, atol=1e-5, rtol=1e-4)
    assert torch.allclose(prof.func_sim.diag(), torch.ones(6), atol=1e-5)


def test_obs_gram_selection_keeps_unique_orthogonal_expert_over_redundant_duplicates():
    """When experts 0..3 are redundant copies and expert 4 is unique/orthogonal, OBS beats freq."""
    torch.manual_seed(3)
    E, N, H = 6, 128, 32
    # Construct active outputs Z[e, n, h]:
    # Experts 0,1,2,3 all output along the SAME direction u0 (high gate mass each, totally redundant!)
    # Expert 4 outputs along orthogonal direction u1 with strong unique energy but slightly lower gate mass
    # Expert 5 outputs along orthogonal direction u2 with strong unique energy
    u0 = F.normalize(torch.randn(N, H, dtype=torch.float64), dim=-1)
    u1 = F.normalize(torch.randn(N, H, dtype=torch.float64), dim=-1)
    u1 = u1 - (u1 * u0).sum(dim=-1, keepdim=True) * u0
    u2 = F.normalize(torch.randn(N, H, dtype=torch.float64), dim=-1)
    u2 = u2 - (u2 * u0).sum(dim=-1, keepdim=True) * u0 - (u2 * u1).sum(dim=-1, keepdim=True) * u1

    Z = torch.stack([1.0 * u0, 1.0 * u0, 1.0 * u0, 1.0 * u0, 1.8 * u1, 1.6 * u2], dim=0)
    G = torch.einsum("enh,fnh->ef", Z, Z)
    # Fake gate_mass favors the 4 redundant copies (0,1,2,3) over unique experts (4,5)
    gate_mass = torch.tensor([100.0, 95.0, 90.0, 85.0, 40.0, 35.0], dtype=torch.float64)
    func_sim = torch.eye(E, dtype=torch.float32)
    for i in range(4):
        for j in range(4):
            func_sim[i, j] = 0.99

    from twyd.moe_compress.profiler import LayerProfile

    prof = LayerProfile(
        layer_idx=0,
        num_experts=E,
        top_k=2,
        norm_topk_prob=False,
        num_tokens=N,
        token_counts=torch.ones(E, dtype=torch.int64) * 50,
        gate_mass=gate_mass,
        output_energy=torch.diagonal(G),
        gram=G,
        func_sim=func_sim,
        teacher_y=Z.sum(dim=0).float(),
        router_probs=torch.full((N, E), 1.0 / E, dtype=torch.bfloat16),
    )

    sel_freq = select_top_k_experts(prof, target_experts=3, method="freq")
    sel_lrc = select_top_k_experts(prof, target_experts=3, method="lrc_cluster")

    # Freq keeps 3 redundant copies [0,1,2]; OBS keeps 1 copy + BOTH unique orthogonal experts 4 & 5!
    assert sel_freq.survivors == [0, 1, 2]
    assert 4 in sel_lrc.survivors and 5 in sel_lrc.survivors
    assert sel_lrc.gram_rel_error < 1e-4
    assert sel_lrc.gram_rel_error < sel_freq.gram_rel_error * 0.05


@pytest.mark.parametrize("norm_topk_prob", [False, True])
@pytest.mark.parametrize("target_top_k", [2, 1])
def test_router_kl_fit_and_lrc_ridge_reduce_val_error(norm_topk_prob: bool, target_top_k: int):
    """Router KL-clone reduces cluster KL, and Coupled LRC Ridge cuts held-out val MSE."""
    model = _tiny_olmoe(seed=19, num_experts=8, top_k=2, norm_topk_prob=norm_topk_prob)
    adapter = OlmoeAdapter()
    moe = model.model.layers[0].mlp
    torch.manual_seed(23)
    x = torch.randn(320, 64)

    prof = profile_moe_layer(adapter, moe, x, layer_idx=0, chunk_size=64, probe_tokens=128)
    sel = select_top_k_experts(prof, target_experts=4, method="lrc_cluster")

    r_res = fit_cluster_router(
        teacher_router_weight=moe.gate.weight,
        survivors=sel.survivors,
        cluster_map=sel.cluster_map,
        x_flat=x,
        teacher_probs=prof.router_probs,
        mode="kl_fit",
        steps=100,
    )
    assert r_res.val_kl_after <= r_res.val_kl_before + 1e-6

    surv_idx = torch.tensor(sel.survivors, dtype=torch.long)
    s_res = solve_lrc_down_proj(
        adapter=adapter,
        x_flat=x,
        y_target=prof.teacher_y,
        router_weight=r_res.router_weight,
        gate_up_proj=moe.experts.gate_up_proj.index_select(0, surv_idx),
        down_proj_init=moe.experts.down_proj.index_select(0, surv_idx),
        act_fn=moe.experts.act_fn,
        top_k=target_top_k,
        norm_topk_prob=norm_topk_prob,
    )
    assert s_res.accepted
    assert s_res.final_val_rel_mse < s_res.init_val_rel_mse


def test_end_to_end_native_hf_export_reload_and_beats_freq_pruning(tmp_path):
    """Full pipeline exports 100% native OlmoeForCausalLM, reloads cleanly, and beats freq pruning."""
    base = _tiny_olmoe(seed=42, num_experts=8, top_k=2, norm_topk_prob=False, num_layers=2)
    # Plant a realistic redundancy pattern + distinct scale so compression has real signal
    with torch.no_grad():
        for layer in base.model.layers:
            # Make expert 1 a near-duplicate of expert 0, expert 3 a near-duplicate of expert 2
            layer.mlp.experts.gate_up_proj[1].copy_(layer.mlp.experts.gate_up_proj[0])
            layer.mlp.experts.down_proj[1].copy_(layer.mlp.experts.down_proj[0])
            layer.mlp.experts.gate_up_proj[3].copy_(layer.mlp.experts.gate_up_proj[2])
            layer.mlp.experts.down_proj[3].copy_(layer.mlp.experts.down_proj[2])

    torch.manual_seed(100)
    calib = [torch.randint(3, 128, (2, 32)) for _ in range(12)]
    holdout = [torch.randint(3, 128, (2, 32)) for _ in range(6)]

    # Baseline: naive frequency pruning + router slice + no LRC ridge
    m_freq = copy.deepcopy(base)
    cfg_freq = MoeCompressConfig(
        target_experts=4,
        target_top_k=2,
        selection_method="freq",
        router_mode="slice",
        use_lrc_ridge=False,
        sequential=True,
    )
    compress_moe_model(m_freq, calib, cfg_freq, verbose=False)
    kl_freq, mse_freq = _teacher_kl_and_mse(base, m_freq, holdout)

    # Ours: LRC-MoE (OBS Gram + functional cluster + KL router + Coupled LRC Ridge)
    m_lrc = copy.deepcopy(base)
    out_dir = str(tmp_path / "olmoe_lrc_e4")
    cfg_lrc = MoeCompressConfig(
        target_experts=4,
        target_top_k=2,
        selection_method="lrc_cluster",
        router_mode="kl_fit",
        use_lrc_ridge=True,
        sequential=True,
    )
    rep = compress_moe_model(m_lrc, calib, cfg_lrc, save_dir=out_dir, verbose=False)
    kl_lrc, mse_lrc = _teacher_kl_and_mse(base, m_lrc, holdout)

    assert rep["params_after"] < rep["params_before"]
    assert rep["mean_final_val_rel_mse"] < rep["mean_init_val_rel_mse"]
    # LRC-MoE must strictly beat naive frequency pruning on held-out teacher fidelity
    assert kl_lrc < kl_freq, f"Expected kl_lrc ({kl_lrc:.6f}) < kl_freq ({kl_freq:.6f})"
    assert mse_lrc < mse_freq, f"Expected mse_lrc ({mse_lrc:.6f}) < mse_freq ({mse_freq:.6f})"

    # Verify 100% native HF reload WITHOUT trust_remote_code
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=False).eval()
    assert isinstance(reloaded, OlmoeForCausalLM)
    assert reloaded.config.num_experts == 4
    assert reloaded.config.num_experts_per_tok == 2

    with torch.no_grad():
        logits_mem = m_lrc(input_ids=holdout[0]).logits
        logits_disk = reloaded(input_ids=holdout[0]).logits
    assert torch.allclose(logits_mem, logits_disk, atol=1e-5, rtol=1e-5)

    # Verify cluster_map saved for Stage 2 Router-KL recovery
    cmap_path = os.path.join(out_dir, "lrc_cluster_map.json")
    assert os.path.isfile(cmap_path)
    with open(cmap_path) as f:
        cmap = json.load(f)
    assert cmap["orig_experts"] == 8 and cmap["new_experts"] == 4
    assert len(cmap["layers"]["0"]["cluster_map"]) == 8


def test_stage2_recovery_logit_kl_router_kl_and_cpu_offloaded_adamw(tmp_path):
    """Verify Stage 2 online KD + Cluster-Router-KL + CpuOffloadedAdamW reduces loss cleanly."""
    from twyd.moe_compress.recover import (
        CpuOffloadedAdamW,
        RecoveryConfig,
        aggregate_teacher_router_targets,
        compute_fused_logit_kl_ce,
        compute_router_kl_loss,
        extract_teacher_topk_targets,
        load_cluster_maps,
        select_trainable_params,
    )

    teacher = _tiny_olmoe(seed=5, num_experts=8, top_k=2, norm_topk_prob=False, num_layers=2)
    student = copy.deepcopy(teacher)
    torch.manual_seed(6)
    calib = [torch.randint(3, 128, (2, 24)) for _ in range(8)]
    out_dir = str(tmp_path / "stage0")
    compress_moe_model(
        student,
        calib,
        MoeCompressConfig(target_experts=4, target_top_k=2, selection_method="lrc_cluster"),
        save_dir=out_dir,
        verbose=False,
    )

    new_e, cmaps, survs = load_cluster_maps(os.path.join(out_dir, "lrc_cluster_map.json"))
    assert new_e == 4 and len(cmaps) == 2 and len(survs) == 2

    cfg = RecoveryConfig(lr=1e-3, kl_topk=32, chunk_size=16, trainable_scope="moe")
    groups = select_trainable_params(student, cfg)
    opt = CpuOffloadedAdamW(groups)

    batch = calib[0]
    labels = batch.clone()
    labels[:, -4:] = -100  # test strict pad masking

    with torch.no_grad():
        t_out = teacher(input_ids=batch, use_cache=False, output_router_logits=True)
        top_idx, logp_t, p_t, p_tail = extract_teacher_topk_targets(t_out.logits, top_k=32, T=1.0, chunk_size=16)
        r_targets = aggregate_teacher_router_targets(t_out.router_logits, cmaps, new_e, survivors=survs, mode="survivor")

    losses = []
    student.train()
    for _ in range(4):
        opt.zero_grad()
        s_out = student(input_ids=batch, use_cache=False, output_router_logits=True)
        kl_l, ce_l = compute_fused_logit_kl_ce(
            s_out.logits, labels, top_idx, logp_t, p_t, p_tail, T=1.0, chunk_size=16, use_checkpoint=False
        )
        rkl_l = compute_router_kl_loss(s_out.router_logits, r_targets)
        loss = 2.0 * kl_l + ce_l + 0.5 * rkl_l
        loss.backward()
        gnorm = opt.step(max_grad_norm=1.0)
        assert gnorm > 0.0
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"Expected recovery loss to decrease: {losses}"


def test_moe_expert_lora_merge_parity(tmp_path):
    """Verify MoeExpertLoRAWrapper Step-0 identity and exact merged native HF reload parity."""
    from twyd.moe_compress.recover import inject_moe_expert_lora_, save_merged_native_moe

    m = _tiny_olmoe(seed=13, num_experts=4, top_k=2, num_layers=2)
    x = torch.randint(3, 128, (2, 16))
    with torch.no_grad():
        y_base = m(input_ids=x, use_cache=False).logits

    n = inject_moe_expert_lora_(m, rank=8, alpha=16.0)
    assert n == 2
    with torch.no_grad():
        y_step0 = m(input_ids=x, use_cache=False).logits
    assert torch.allclose(y_base, y_step0, atol=1e-6, rtol=1e-6)

    # Perturb B matrices to simulate training, then verify merged native reload matches wrapped forward
    with torch.no_grad():
        for layer in m.model.layers:
            layer.mlp.experts.gu_B.normal_(0.0, 0.05)
            layer.mlp.experts.dn_B.normal_(0.0, 0.05)
        y_wrapped = m(input_ids=x, use_cache=False).logits

    save_dir = str(tmp_path / "merged_native")
    save_merged_native_moe(m, save_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=False).eval()
    assert isinstance(reloaded, OlmoeForCausalLM)
    with torch.no_grad():
        y_disk = reloaded(input_ids=x, use_cache=False).logits
    assert torch.allclose(y_wrapped, y_disk, atol=1e-4, rtol=1e-4)


def test_hidden_cosine_alignment_jsd_and_all_no_embed_scope():
    """Verify same-width hidden cosine alignment, Generalized JSD, and all_no_embed scope."""
    from twyd.moe_compress.recover import (
        RecoveryConfig,
        compute_fused_logit_kl_ce,
        compute_hidden_cosine_loss,
        extract_teacher_hidden_targets,
        extract_teacher_topk_targets,
        select_trainable_params,
    )

    teacher = _tiny_olmoe(seed=31, num_experts=8, top_k=2, num_layers=2)
    student = _tiny_olmoe(seed=32, num_experts=4, top_k=2, num_layers=2)
    x = torch.randint(3, 128, (2, 16))
    labels = x.clone()
    labels[:, :6] = -100  # prompt mask

    with torch.no_grad():
        t_out = teacher(input_ids=x, use_cache=False, output_hidden_states=True)
        top_idx, logp_t, p_t, p_tail = extract_teacher_topk_targets(t_out.logits, 32, 1.0, 16)
        h_targets = extract_teacher_hidden_targets(t_out.hidden_states, (0, 1))
    assert len(h_targets) == 2

    cfg = RecoveryConfig(trainable_scope="all_no_embed", attn_lr_scale=0.3, router_lr_scale=0.5)
    groups = select_trainable_params(student, cfg)
    assert len(groups) == 3  # main (MoE), attn, router
    assert not student.model.embed_tokens.weight.requires_grad
    assert not student.lm_head.weight.requires_grad
    assert student.model.layers[0].self_attn.q_proj.weight.requires_grad

    s_out = student(input_ids=x, use_cache=False, output_hidden_states=True)
    jsd_l, ce_l = compute_fused_logit_kl_ce(
        s_out.logits, labels, top_idx, logp_t, p_t, p_tail, T=1.0, chunk_size=16, use_checkpoint=False, jsd_beta=0.3
    )
    hid_l = compute_hidden_cosine_loss(s_out.hidden_states, h_targets, (0, 1), token_mask=(labels != -100))
    assert float(jsd_l.item()) > 0.0 and float(ce_l.item()) > 0.0 and float(hid_l.item()) > 0.0
    (jsd_l + ce_l + 0.5 * hid_l).backward()
    assert student.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_fast_block_dispatch_parity_and_fullrank_nonlinear_distill():
    """Verify single-argsort dispatch matches native OlmoeSparseMoeBlock and cuts valMSE sharply."""
    from twyd.moe_compress.block_distill import (
        BlockDistillConfig,
        distill_moe_layer_fullrank,
        fast_olmoe_block_forward,
    )

    teacher = _tiny_olmoe(seed=71, num_experts=8, top_k=2, norm_topk_prob=False, num_layers=1)
    student = _tiny_olmoe(seed=72, num_experts=4, top_k=2, norm_topk_prob=False, num_layers=1)
    s_mlp = student.model.layers[0].mlp

    torch.manual_seed(73)
    x = torch.randn(64, 64)
    # 1. Exact forward parity vs native OlmoeSparseMoeBlock
    with torch.no_grad():
        ref_y = s_mlp(x.unsqueeze(0)).squeeze(0)
        fast_y, _, aux = fast_olmoe_block_forward(
            x, s_mlp.gate.weight, s_mlp.experts.gate_up_proj, s_mlp.experts.down_proj, s_mlp.experts.act_fn, 2, False
        )
    assert torch.allclose(ref_y, fast_y, atol=1e-5, rtol=1e-4)
    assert float(aux.item()) > 0.0

    # 2. Full-rank nonlinear distillation strictly reduces valMSE against 8-expert teacher
    x_big = torch.randn(512, 64)
    with torch.no_grad():
        y_teacher = teacher.model.layers[0].mlp(x_big.unsqueeze(0)).squeeze(0)

    cfg = BlockDistillConfig(steps_per_layer=80, batch_tokens=128, lr=5e-3, eval_every=40, warmup_steps=5)
    st = distill_moe_layer_fullrank(s_mlp, x_big, y_teacher, cfg, layer_idx=0, device="cpu", verbose=False)
    assert st.best_val_rel_mse < st.init_val_rel_mse * 0.75, (
        f"Expected sharp valMSE reduction: {st.init_val_rel_mse:.4f} -> {st.best_val_rel_mse:.4f}"
    )




