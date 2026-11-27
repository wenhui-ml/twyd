"""Architecture adapters for native HuggingFace MoE models.

Each adapter exposes a uniform interface to inspect router weights, stacked
expert weights (gate_up_proj / down_proj), exact routing semantics, and
in-place native rebuild with updated `num_experts` / `num_experts_per_tok`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdapter(ABC):
    """Uniform interface over native HF Sparse-MoE blocks."""

    @abstractmethod
    def iter_moe_layers(self, model: nn.Module) -> Iterator[Tuple[int, nn.Module, nn.Module]]:
        """Yield `(layer_idx, decoder_layer, moe_block)` for every MoE layer."""

    @abstractmethod
    def get_num_experts(self, moe_block: nn.Module) -> int:
        ...

    @abstractmethod
    def get_top_k(self, moe_block: nn.Module) -> int:
        ...

    @abstractmethod
    def get_norm_topk_prob(self, moe_block: nn.Module) -> bool:
        ...

    @abstractmethod
    def get_router_weight(self, moe_block: nn.Module) -> torch.Tensor:
        """Return router weight of shape `(E, H)`."""

    @abstractmethod
    def get_expert_weights(
        self, moe_block: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
        """Return `(gate_up_proj[E, 2*d_ff, H], down_proj[E, H, d_ff], act_fn)`."""

    @abstractmethod
    def route_with_weight(
        self,
        x_flat: torch.Tensor,
        router_weight: torch.Tensor,
        top_k: int,
        norm_topk_prob: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(router_logits[N,E], router_probs[N,E], top_weights[N,k], top_idx[N,k])`."""

    @abstractmethod
    def expert_hidden(
        self,
        x: torch.Tensor,
        gate_up_w: torch.Tensor,
        act_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute intermediate SwiGLU activation `act(x W_gate^T) * (x W_up^T)`."""

    @abstractmethod
    def expert_forward(
        self,
        x: torch.Tensor,
        gate_up_w: torch.Tensor,
        down_w: torch.Tensor,
        act_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute single expert output `(N, H)`."""

    @abstractmethod
    def rebuild_moe_block_(
        self,
        model: nn.Module,
        layer_idx: int,
        new_router_weight: torch.Tensor,
        new_gate_up_proj: torch.Tensor,
        new_down_proj: torch.Tensor,
        new_num_experts: int,
        new_top_k: int,
    ) -> None:
        """Replace layer `layer_idx`'s MoE block in-place with native `new_num_experts` tensors."""

    @abstractmethod
    def finalize_config_(
        self,
        model: nn.Module,
        new_num_experts: int,
        new_top_k: int,
    ) -> None:
        """Update `model.config` in-place so `save_pretrained` exports native HF config."""


class OlmoeAdapter(MoEAdapter):
    """Adapter for HuggingFace `OlmoeForCausalLM` (transformers >= 5.x fused 3D OlmoeExperts)."""

    def iter_moe_layers(self, model: nn.Module) -> Iterator[Tuple[int, nn.Module, nn.Module]]:
        layers = model.model.layers
        for idx, layer in enumerate(layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
                yield idx, layer, layer.mlp

    def get_num_experts(self, moe_block: nn.Module) -> int:
        return int(moe_block.gate.num_experts)

    def get_top_k(self, moe_block: nn.Module) -> int:
        return int(moe_block.gate.top_k)

    def get_norm_topk_prob(self, moe_block: nn.Module) -> bool:
        return bool(moe_block.gate.norm_topk_prob)

    def get_router_weight(self, moe_block: nn.Module) -> torch.Tensor:
        return moe_block.gate.weight

    def get_expert_weights(
        self, moe_block: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
        experts = moe_block.experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            return experts.gate_up_proj, experts.down_proj, experts.act_fn
        raise TypeError(
            f"Unsupported OlmoeExperts layout: {type(experts).__name__}; "
            "expected fused 3D gate_up_proj and down_proj tensors."
        )

    def route_with_weight(
        self,
        x_flat: torch.Tensor,
        router_weight: torch.Tensor,
        top_k: int,
        norm_topk_prob: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(x_flat, router_weight.to(dtype=x_flat.dtype, device=x_flat.device))
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
        if norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return logits, probs, top_vals.to(logits.dtype), top_idx

    def expert_hidden(
        self,
        x: torch.Tensor,
        gate_up_w: torch.Tensor,
        act_fn: nn.Module,
    ) -> torch.Tensor:
        gate, up = F.linear(x, gate_up_w).chunk(2, dim=-1)
        return act_fn(gate) * up

    def expert_forward(
        self,
        x: torch.Tensor,
        gate_up_w: torch.Tensor,
        down_w: torch.Tensor,
        act_fn: nn.Module,
    ) -> torch.Tensor:
        h = self.expert_hidden(x, gate_up_w, act_fn)
        return F.linear(h, down_w)

    def rebuild_moe_block_(
        self,
        model: nn.Module,
        layer_idx: int,
        new_router_weight: torch.Tensor,
        new_gate_up_proj: torch.Tensor,
        new_down_proj: torch.Tensor,
        new_num_experts: int,
        new_top_k: int,
    ) -> None:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        dev = mlp.gate.weight.device
        dt = mlp.gate.weight.dtype

        mlp.gate.num_experts = int(new_num_experts)
        mlp.gate.top_k = int(new_top_k)
        mlp.gate.weight = nn.Parameter(
            new_router_weight.detach().to(device=dev, dtype=dt).contiguous()
        )

        mlp.experts.num_experts = int(new_num_experts)
        mlp.experts.gate_up_proj = nn.Parameter(
            new_gate_up_proj.detach().to(device=dev, dtype=dt).contiguous()
        )
        mlp.experts.down_proj = nn.Parameter(
            new_down_proj.detach().to(device=dev, dtype=dt).contiguous()
        )

    def finalize_config_(
        self,
        model: nn.Module,
        new_num_experts: int,
        new_top_k: int,
    ) -> None:
        model.config.num_experts = int(new_num_experts)
        if hasattr(model.config, "num_local_experts"):
            try:
                model.config.num_local_experts = int(new_num_experts)
            except AttributeError:
                pass
        model.config.num_experts_per_tok = int(new_top_k)
        # OlmoeForCausalLM caches num_experts / num_experts_per_tok as instance attributes
        # used by load_balancing_loss_func(output_router_logits=True).
        for obj in (model, getattr(model, "model", None)):
            if obj is None:
                continue
            if hasattr(obj, "num_experts"):
                obj.num_experts = int(new_num_experts)
            if hasattr(obj, "num_experts_per_tok"):
                obj.num_experts_per_tok = int(new_top_k)


def get_moe_adapter(model: nn.Module) -> MoEAdapter:
    """Resolve the appropriate MoEAdapter for `model`."""
    mtype = getattr(getattr(model, "config", None), "model_type", "")
    if mtype == "olmoe":
        return OlmoeAdapter()
    raise ValueError(
        f"Unsupported MoE model_type={mtype!r}. Currently supported: 'olmoe'."
    )
