"""LRC-MoE: Low-Rank Clone Expert Compression for native HuggingFace MoE models.

Compresses an E-expert MoE teacher into a K'-expert (and optionally narrower
active top-m') native HF MoE student (zero custom runtime ops, zero
trust_remote_code) via:
  1. Calibration functional profiling (exact E x E active output Gram matrix +
     permutation-invariant functional probe similarity).
  2. Redundancy-aware Top-K' selection (exact OBS/Schur backward elimination +
     1-opt swap polishing) and functional clustering of pruned experts.
  3. Convex probability-mass-preserving Router KL-Clone.
  4. Coupled Gauss-Seidel LRC Ridge absorption into surviving experts'
     down_proj weights with held-out calibration auto-damping.
"""
from .adapters import MoEAdapter, OlmoeAdapter, get_moe_adapter
from .lrc_solver import (
    RouterFitResult,
    SolverResult,
    fit_cluster_router,
    solve_lrc_down_proj,
)
from .pipeline import LayerCompressionStats, MoeCompressConfig, compress_moe_model
from .profiler import LayerProfile, profile_moe_layer
from .recover import (
    CpuOffloadedAdamW,
    RecoveryConfig,
    aggregate_teacher_router_targets,
    compute_fused_logit_kl_ce,
    compute_router_kl_loss,
    extract_teacher_topk_targets,
    load_cluster_maps,
    select_trainable_params,
)
from .selector import SelectionResult, select_top_k_experts

__all__ = [
    "MoEAdapter",
    "OlmoeAdapter",
    "get_moe_adapter",
    "LayerProfile",
    "profile_moe_layer",
    "SelectionResult",
    "select_top_k_experts",
    "RouterFitResult",
    "SolverResult",
    "fit_cluster_router",
    "solve_lrc_down_proj",
    "MoeCompressConfig",
    "LayerCompressionStats",
    "compress_moe_model",
    "RecoveryConfig",
    "CpuOffloadedAdamW",
    "load_cluster_maps",
    "extract_teacher_topk_targets",
    "aggregate_teacher_router_targets",
    "compute_fused_logit_kl_ce",
    "compute_router_kl_loss",
    "select_trainable_params",
]
