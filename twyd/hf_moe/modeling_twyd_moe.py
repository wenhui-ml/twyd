"""Self-contained HF Model for TWYD-MoE (Shared-Dense Mother + Routed Residual Experts).

At step 0, every routed expert's down_proj is zero-initialized, so the model is
bit-exact identical to the dense mother (zero warm-start degradation) while
expanding static factual knowledge capacity without increasing hidden_size r.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaMLP

try:
    from .configuration_twyd_moe import TwydMoEConfig
except ImportError:
    from configuration_twyd_moe import TwydMoEConfig


class _AttachAuxLoss(torch.autograd.Function):
    """Attach scalar MoE load-balance gradient cleanly through gradient checkpointing."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, aux_loss: torch.Tensor, aux_scale: float = 1.0):
        ctx.aux_dtype = aux_loss.dtype
        ctx.aux_scale = float(aux_scale)
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        return grad_x, torch.full((), ctx.aux_scale, device=grad_x.device, dtype=ctx.aux_dtype), None


class TwydSharedRoutedMoE(nn.Module):
    """y = shared_mlp(x) + sum_{e in TopK(x)} g_e(x) * Expert_e(x)."""

    def __init__(
        self,
        shared_mlp: nn.Module,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        lb_coef: float = 0.02,
        z_coef: float = 0.001,
    ):
        super().__init__()
        self.shared_mlp = shared_mlp
        self.hidden_size = int(hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.lb_coef = float(lb_coef)
        self.z_coef = float(z_coef)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.w_gate = nn.Parameter(torch.empty(self.num_experts, self.moe_intermediate_size, self.hidden_size))
        self.w_up = nn.Parameter(torch.empty(self.num_experts, self.moe_intermediate_size, self.hidden_size))
        self.w_down = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, self.moe_intermediate_size))

        self.last_aux_val: float = 0.0
        self.last_load: torch.Tensor | None = None
        self.aux_grad_scale: float = 1.0
        self.reset_routed_parameters()

    def reset_routed_parameters(self) -> None:
        with torch.no_grad():
            # Orthogonal rows on router so tokens spread across all experts at step 0.
            nn.init.orthogonal_(self.router.weight)
            self.router.weight.mul_(0.02)
            # Orthogonal subspaces per expert; orthogonal_ already gives row L2 norm 1
            # when d_e <= hidden (or sqrt(hidden/d_e) when d_e > hidden). Scaling by
            # gain=0.8*sqrt(max(d_e, hidden)/hidden) matches Llama's ~0.02 entry std
            # so silu(gate)*up is O(1) and w_down gets strong gradients from step 1.
            gain = 0.8 * math.sqrt(max(self.moe_intermediate_size, self.hidden_size) / self.hidden_size)
            for e in range(self.num_experts):
                nn.init.orthogonal_(self.w_gate[e], gain=gain)
                nn.init.orthogonal_(self.w_up[e], gain=gain)
            # Zero down_proj => exact step-0 identity with the dense mother!
            self.w_down.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.shared_mlp(x)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        n_tok = x_flat.shape[0]
        if n_tok == 0:
            self.last_aux_val = 0.0
            return base_out

        router_logits = F.linear(x_flat.float(), self.router.weight.float())  # [N, E]
        probs = F.softmax(router_logits, dim=-1)
        topk_w, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Switch load-balance loss + router z-loss (computed in fp32).
        # Detach x_flat for aux_loss so load-balancing trains this layer's router
        # without distorting earlier layers' shared hidden representations.
        aux_loss = None
        if self.training:
            aux_logits = F.linear(x_flat.detach().float(), self.router.weight.float())
            aux_probs = F.softmax(aux_logits, dim=-1)
            one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # [N, K, E]
            f_e = one_hot.mean(dim=(0, 1))  # fraction of routed slots per expert
            p_e = aux_probs.mean(dim=0)     # mean router probability per expert
            lb_loss = self.num_experts * (f_e * p_e).sum()
            z_loss = torch.logsumexp(aux_logits, dim=-1).pow(2).mean()
            aux_loss = self.lb_coef * lb_loss + self.z_coef * z_loss
            self.last_aux_val = float(aux_loss.detach())
            self.last_load = f_e.detach()

        # Sort tokens by expert once per layer (1 tiny bincount sync instead of E torch.where syncs).
        flat_e = topk_idx.reshape(-1)
        flat_w = topk_w.reshape(-1).to(x_flat.dtype)
        order = torch.argsort(flat_e)
        tok_order = torch.div(order, self.top_k, rounding_mode="floor")
        x_sorted = x_flat.index_select(0, tok_order)
        counts = torch.bincount(flat_e, minlength=self.num_experts).tolist()
        chunks = torch.split(x_sorted, counts, dim=0)
        ys = []
        for e, xe in enumerate(chunks):
            if counts[e] == 0:
                continue
            he = F.silu(F.linear(xe, self.w_gate[e])) * F.linear(xe, self.w_up[e])
            ys.append(F.linear(he, self.w_down[e]))
        delta_flat = torch.zeros_like(x_flat)
        if ys:
            y_sorted = torch.cat(ys, dim=0) * flat_w.index_select(0, order).unsqueeze(-1)
            delta_flat.index_add_(0, tok_order, y_sorted)

        out = base_out + delta_flat.reshape(orig_shape)
        if aux_loss is not None and torch.is_grad_enabled() and aux_loss.requires_grad:
            out = _AttachAuxLoss.apply(out, aux_loss, self.aux_grad_scale)
        return out


def inject_twyd_moe_(
    model: nn.Module,
    moe_layers: list[int],
    num_experts: int = 8,
    top_k: int = 2,
    moe_intermediate_size: int = 1536,
    lb_coef: float = 0.02,
    z_coef: float = 0.001,
) -> list[int]:
    """Wrap selected decoder layers' MLPs with TwydSharedRoutedMoE in-place."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Cannot locate decoder layers on model")
    hidden_size = int(getattr(model.config, "hidden_size"))
    injected = []
    for li in sorted(set(int(i) for i in moe_layers)):
        if not (0 <= li < len(layers)):
            continue
        cur_mlp = layers[li].mlp
        if isinstance(cur_mlp, TwydSharedRoutedMoE):
            injected.append(li)
            continue
        ref = next(cur_mlp.parameters())
        moe_mod = TwydSharedRoutedMoE(
            shared_mlp=cur_mlp,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            lb_coef=lb_coef,
            z_coef=z_coef,
        ).to(device=ref.device, dtype=ref.dtype)
        layers[li].mlp = moe_mod
        injected.append(li)
    return injected


class TwydMoEForCausalLM(LlamaForCausalLM):
    config_class = TwydMoEConfig

    def __init__(self, config: TwydMoEConfig):
        super().__init__(config)
        if getattr(config, "moe_layers", None):
            inject_twyd_moe_(
                self,
                moe_layers=list(config.moe_layers),
                num_experts=int(getattr(config, "num_experts", 8)),
                top_k=int(getattr(config, "num_experts_per_tok", 2)),
                moe_intermediate_size=int(getattr(config, "moe_intermediate_size", 1536)),
                lb_coef=float(getattr(config, "moe_lb_coef", 0.02)),
                z_coef=float(getattr(config, "moe_z_coef", 0.001)),
            )
            self.post_init()
            # Re-zero w_down after post_init() in case HF weight init touched it.
            with torch.no_grad():
                for mod in self.modules():
                    if isinstance(mod, TwydSharedRoutedMoE):
                        mod.w_down.zero_()
