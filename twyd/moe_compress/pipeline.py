"""End-to-end Step-0 LRC-MoE Top-K' expert compression pipeline.

Captures real per-layer MoE calibration inputs cleanly via forward hooks (or
sequential error-compensated propagation) and compresses every MoE layer in
place to `target_experts` (and optional `target_top_k`), producing a 100%
native HF MoE model + `lrc_cluster_map.json` for Stage 2 Router-KL recovery.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .adapters import get_moe_adapter
from .lrc_solver import fit_cluster_router, solve_lrc_down_proj
from .profiler import profile_moe_layer
from .selector import select_top_k_experts


@dataclass
class MoeCompressConfig:
    target_experts: int
    target_top_k: Optional[int] = None   # None -> keep original (clamped to target_experts)
    selection_method: str = "lrc_cluster"  # freq | energy | gram_greedy | lrc_cluster
    router_mode: str = "kl_fit"            # slice | kl_fit
    use_lrc_ridge: bool = True
    lrc_rank: int = 0                      # 0 = full-rank damped ridge; >0 = low-rank SVD delta
    sequential: bool = True                # sequential input-compensated calibration
    probe_tokens: int = 512
    chunk_size: int = 512
    val_frac: float = 0.15
    router_steps: int = 120
    router_lr: float = 5e-2
    router_l2_reg: float = 1e-4
    ridge_sweeps: int = 3
    ridge_step: float = 0.85
    ridge_alphas: Tuple[float, ...] = (0.01, 0.1, 0.5, 2.0, 10.0)
    gram_weight: float = 0.25


@dataclass
class LayerCompressionStats:
    layer_idx: int
    orig_experts: int
    new_experts: int
    orig_top_k: int
    new_top_k: int
    selection_method: str
    router_mode: str
    use_lrc_ridge: bool
    survivors: List[int]
    cluster_map: List[int]
    gram_rel_error: float
    retained_energy_ratio: float
    retained_mass_ratio: float
    router_val_kl_before: float
    router_val_kl_after: float
    init_val_rel_mse: float
    final_val_rel_mse: float
    best_alpha: float
    elapsed_sec: float


def _collect_layer_moe_inputs(
    model: nn.Module,
    moe_block: nn.Module,
    calib_batches: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Run calibration batches and capture exact 2D inputs `(N, H)` to `moe_block`."""
    captured: List[torch.Tensor] = []

    def _pre_hook(_mod, args):
        h = args[0]
        captured.append(h.detach().reshape(-1, h.shape[-1]).cpu())

    handle = moe_block.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            for batch in calib_batches:
                ids = batch.to(device=device)
                model.model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()

    return torch.cat(captured, dim=0)


def _capture_all_moe_inputs(
    model: nn.Module,
    moe_layers: List[Tuple[int, nn.Module, nn.Module]],
    calib_batches: Sequence[torch.Tensor],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Capture inputs to ALL MoE layers in a single forward pass over `calib_batches`."""
    captured: Dict[int, List[torch.Tensor]] = {idx: [] for idx, _, _ in moe_layers}
    handles = []

    for idx, _, moe_block in moe_layers:
        def _make_hook(layer_id: int):
            def _pre_hook(_mod, args):
                h = args[0]
                captured[layer_id].append(h.detach().reshape(-1, h.shape[-1]).cpu())
            return _pre_hook
        handles.append(moe_block.register_forward_pre_hook(_make_hook(idx)))

    try:
        with torch.no_grad():
            for batch in calib_batches:
                ids = batch.to(device=device)
                model.model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {idx: torch.cat(chunks, dim=0) for idx, chunks in captured.items()}


def _run_layer_forward_sequential(
    model: nn.Module,
    calib_batches: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
    """Run embeddings + RoPE once so layers 0..L-1 can execute strictly sequentially (O(L) total)."""
    hidden_list: List[torch.Tensor] = []
    pos_emb_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    pos_id_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in calib_batches:
            ids = batch.to(device=device)
            h = model.model.embed_tokens(ids)
            pos_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0).expand_as(ids)
            pos_emb = model.model.rotary_emb(h, pos_ids)
            hidden_list.append(h.cpu())
            pos_emb_list.append((pos_emb[0].cpu(), pos_emb[1].cpu()))
            pos_id_list.append(pos_ids.cpu())

    return hidden_list, pos_emb_list, pos_id_list


def _layer_pre_mlp_inputs(
    layer: nn.Module,
    hidden_list: List[torch.Tensor],
    pos_emb_list: List[Tuple[torch.Tensor, torch.Tensor]],
    pos_id_list: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Capture exact inputs entering `layer.mlp` without re-running earlier layers."""
    captured: List[torch.Tensor] = []

    def _pre_hook(_mod, args):
        h = args[0]
        captured.append(h.detach().reshape(-1, h.shape[-1]).cpu())

    handle = layer.mlp.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            for h_cpu, (cos_c, sin_c), pid_c in zip(hidden_list, pos_emb_list, pos_id_list):
                h = h_cpu.to(device=device)
                pe = (cos_c.to(device=device), sin_c.to(device=device))
                pid = pid_c.to(device=device)
                layer(h, position_embeddings=pe, position_ids=pid)
    finally:
        handle.remove()

    return torch.cat(captured, dim=0)


def _advance_hidden_states(
    layer: nn.Module,
    hidden_list: List[torch.Tensor],
    pos_emb_list: List[Tuple[torch.Tensor, torch.Tensor]],
    pos_id_list: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Propagate `hidden_list` forward through the (now compressed) `layer`."""
    next_hidden: List[torch.Tensor] = []
    with torch.no_grad():
        for h_cpu, (cos_c, sin_c), pid_c in zip(hidden_list, pos_emb_list, pos_id_list):
            h = h_cpu.to(device=device)
            pe = (cos_c.to(device=device), sin_c.to(device=device))
            pid = pid_c.to(device=device)
            out = layer(h, position_embeddings=pe, position_ids=pid)
            if isinstance(out, tuple):
                out = out[0]
            next_hidden.append(out.detach().cpu())
    return next_hidden


def compress_moe_model(
    model: nn.Module,
    calib_batches: Sequence[torch.Tensor],
    cfg: MoeCompressConfig,
    device: torch.device | str | None = None,
    save_dir: Optional[str] = None,
    tokenizer=None,
    verbose: bool = True,
) -> Dict:
    """Compress all MoE layers in `model` in-place from E -> `cfg.target_experts`.

    Returns a JSON-serializable report dict containing per-layer metrics and
    the full `cluster_map` needed for Stage 2 Router-KL recovery distillation.
    """
    t_total = time.time()
    model.eval()
    adapter = get_moe_adapter(model)
    moe_layers = list(adapter.iter_moe_layers(model))
    if not moe_layers:
        raise ValueError("No MoE layers found in model.")

    dev = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )

    orig_e = adapter.get_num_experts(moe_layers[0][2])
    orig_k = adapter.get_top_k(moe_layers[0][2])
    new_e = int(cfg.target_experts)
    new_k = min(int(cfg.target_top_k) if cfg.target_top_k is not None else orig_k, new_e)

    params_before = sum(p.numel() for p in model.parameters())

    # Choose fast O(L) sequential propagation vs single-pass parallel capture
    if cfg.sequential:
        hidden_list, pos_emb_list, pos_id_list = _run_layer_forward_sequential(
            model, calib_batches, dev
        )
        parallel_inputs = None
    else:
        hidden_list = pos_emb_list = pos_id_list = None
        parallel_inputs = _capture_all_moe_inputs(model, moe_layers, calib_batches, dev)

    layer_stats: List[LayerCompressionStats] = []
    cluster_maps: Dict[str, Dict] = {}

    for layer_idx, layer_mod, moe_block in moe_layers:
        t0 = time.time()
        if cfg.sequential:
            x_flat = _layer_pre_mlp_inputs(
                layer_mod, hidden_list, pos_emb_list, pos_id_list, dev
            )
        else:
            x_flat = parallel_inputs[layer_idx]

        profile = profile_moe_layer(
            adapter=adapter,
            moe_block=moe_block,
            x_flat=x_flat,
            layer_idx=layer_idx,
            chunk_size=cfg.chunk_size,
            probe_tokens=cfg.probe_tokens,
            device=dev,
        )

        sel = select_top_k_experts(
            profile=profile,
            target_experts=new_e,
            method=cfg.selection_method,
            gram_weight=cfg.gram_weight,
        )

        router_w = adapter.get_router_weight(moe_block)
        gate_up_proj, down_proj, act_fn = adapter.get_expert_weights(moe_block)
        norm_topk = adapter.get_norm_topk_prob(moe_block)

        router_res = fit_cluster_router(
            teacher_router_weight=router_w,
            survivors=sel.survivors,
            cluster_map=sel.cluster_map,
            x_flat=x_flat,
            teacher_probs=profile.router_probs,
            mode=cfg.router_mode,
            val_frac=cfg.val_frac,
            l2_reg=cfg.router_l2_reg,
            lr=cfg.router_lr,
            steps=cfg.router_steps,
            device=dev,
        )

        surv_idx = torch.tensor(sel.survivors, dtype=torch.long, device=gate_up_proj.device)
        new_gate_up = gate_up_proj.index_select(0, surv_idx).detach().clone()
        new_down_init = down_proj.index_select(0, surv_idx).detach().clone()

        if cfg.use_lrc_ridge:
            solve_res = solve_lrc_down_proj(
                adapter=adapter,
                x_flat=x_flat,
                y_target=profile.teacher_y,
                router_weight=router_res.router_weight,
                gate_up_proj=new_gate_up,
                down_proj_init=new_down_init,
                act_fn=act_fn,
                top_k=new_k,
                norm_topk_prob=norm_topk,
                val_frac=cfg.val_frac,
                alphas=cfg.ridge_alphas,
                n_sweeps=cfg.ridge_sweeps,
                damping_step=cfg.ridge_step,
                lrc_rank=cfg.lrc_rank,
                device=dev,
            )
            new_down = solve_res.down_proj
            init_val_mse = solve_res.init_val_rel_mse
            final_val_mse = solve_res.final_val_rel_mse
            best_alpha = solve_res.best_alpha
        else:
            # Evaluate init val MSE for apple-to-apple logging
            solve_res = solve_lrc_down_proj(
                adapter=adapter,
                x_flat=x_flat,
                y_target=profile.teacher_y,
                router_weight=router_res.router_weight,
                gate_up_proj=new_gate_up,
                down_proj_init=new_down_init,
                act_fn=act_fn,
                top_k=new_k,
                norm_topk_prob=norm_topk,
                val_frac=cfg.val_frac,
                alphas=(),
                n_sweeps=0,
                device=dev,
            )
            new_down = new_down_init
            init_val_mse = solve_res.init_val_rel_mse
            final_val_mse = solve_res.init_val_rel_mse
            best_alpha = float("inf")

        adapter.rebuild_moe_block_(
            model=model,
            layer_idx=layer_idx,
            new_router_weight=router_res.router_weight,
            new_gate_up_proj=new_gate_up,
            new_down_proj=new_down,
            new_num_experts=new_e,
            new_top_k=new_k,
        )

        # Advance student hidden states through the newly compressed layer
        if cfg.sequential:
            hidden_list = _advance_hidden_states(
                layer_mod, hidden_list, pos_emb_list, pos_id_list, dev
            )

        elapsed = time.time() - t0
        st = LayerCompressionStats(
            layer_idx=layer_idx,
            orig_experts=orig_e,
            new_experts=new_e,
            orig_top_k=orig_k,
            new_top_k=new_k,
            selection_method=cfg.selection_method,
            router_mode=cfg.router_mode,
            use_lrc_ridge=cfg.use_lrc_ridge,
            survivors=sel.survivors,
            cluster_map=sel.cluster_map,
            gram_rel_error=sel.gram_rel_error,
            retained_energy_ratio=sel.retained_energy_ratio,
            retained_mass_ratio=sel.retained_mass_ratio,
            router_val_kl_before=router_res.val_kl_before,
            router_val_kl_after=router_res.val_kl_after,
            init_val_rel_mse=init_val_mse,
            final_val_rel_mse=final_val_mse,
            best_alpha=best_alpha,
            elapsed_sec=elapsed,
        )
        layer_stats.append(st)
        cluster_maps[str(layer_idx)] = {
            "survivors": sel.survivors,
            "cluster_map": sel.cluster_map,
        }

        if verbose:
            print(
                f"[lrc-moe] L{layer_idx:02d} {orig_e}->{new_e} (top{orig_k}->top{new_k}) | "
                f"retE={sel.retained_energy_ratio:.3f} gramErr={sel.gram_rel_error:.4f} | "
                f"rKL={router_res.val_kl_before:.4f}->{router_res.val_kl_after:.4f} | "
                f"valMSE={init_val_mse:.4f}->{final_val_mse:.4f} (a={best_alpha}) | {elapsed:.1f}s",
                flush=True,
            )

        if dev.type == "cuda":
            torch.cuda.empty_cache()

    adapter.finalize_config_(model, new_e, new_k)
    params_after = sum(p.numel() for p in model.parameters())

    report = {
        "config": asdict(cfg),
        "orig_experts": orig_e,
        "new_experts": new_e,
        "orig_top_k": orig_k,
        "new_top_k": new_k,
        "params_before": params_before,
        "params_after": params_after,
        "param_ratio": params_after / max(params_before, 1),
        "mean_init_val_rel_mse": sum(s.init_val_rel_mse for s in layer_stats) / len(layer_stats),
        "mean_final_val_rel_mse": sum(s.final_val_rel_mse for s in layer_stats) / len(layer_stats),
        "mean_router_val_kl_before": sum(s.router_val_kl_before for s in layer_stats) / len(layer_stats),
        "mean_router_val_kl_after": sum(s.router_val_kl_after for s in layer_stats) / len(layer_stats),
        "total_elapsed_sec": time.time() - t_total,
        "layers": [asdict(s) for s in layer_stats],
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "lrc_cluster_map.json"), "w") as f:
            json.dump(
                {
                    "orig_experts": orig_e,
                    "new_experts": new_e,
                    "orig_top_k": orig_k,
                    "new_top_k": new_k,
                    "layers": cluster_maps,
                },
                f,
                indent=2,
            )
        with open(os.path.join(save_dir, "lrc_compression_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    return report
