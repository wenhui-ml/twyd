"""Self-contained HF Config for TWYD-MoE (r=1536 Shared-Dense + Routed Knowledge Experts)."""
from transformers.models.llama.configuration_llama import LlamaConfig


class TwydMoEConfig(LlamaConfig):
    model_type = "twyd_moe"

    def __init__(
        self,
        moe_layers=None,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 1536,
        moe_lb_coef: float = 0.02,
        moe_z_coef: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moe_layers = list(moe_layers) if moe_layers is not None else []
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.moe_lb_coef = float(moe_lb_coef)
        self.moe_z_coef = float(moe_z_coef)
