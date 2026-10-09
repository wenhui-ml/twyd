"""TWYD LRC distillation config."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TwydLRCConfig:
    """Config shared by tuner / loss / init / merge.

    branch: "dense" | "core" | "plain" (plain = matched LRC baseline, col path only).
    teacher_model: HF id or local path of the teacher (frozen, for KL + SVD basis).
    student_hidden: r = d_model^(S), e.g. 1536 for Llama3.2-3B->1.5B.
    target_modules: MLP projections to open. Default gate/up/down only (paper App. B).
    kl_temperature: T=40 (paper A.5). kl_weight=wKL=1.0, ntp_weight=wNTP=1.0.
    aux_weight: lambda_aux=0.2. use_aux toggles hidden/attn/mlp alignment.
    aux_loss_type: "mseloss" | "l1".
    ridge_lambda: None -> auto tr(C^TC)/r (Eq.4); else explicit.
    outlier_keep: k outlier passthrough rows for Qwen P (Eq.5, default 0 = plain SVD).
    """

    branch: str = "core"
    teacher_model: Optional[str] = None
    student_hidden: Optional[int] = None
    target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    kl_temperature: float = 40.0
    kl_weight: float = 1.0
    ntp_weight: float = 1.0
    aux_weight: float = 0.2
    use_aux: bool = True
    aux_loss_type: str = "mseloss"
    ridge_lambda: Optional[float] = None
    outlier_keep: int = 0

    def validate(self) -> None:
        assert self.branch in ("dense", "core", "plain"), f"branch={self.branch}"
        assert self.aux_loss_type in ("mseloss", "l1"), self.aux_loss_type
