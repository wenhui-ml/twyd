"""TWYD LRC distillation package (ms-swift plugins)."""
from .config import TwydLRCConfig
from .core_basis import build_core_basis, check_basis
from .init import build_outlier_aware_P, ridge_zcol, zero_perp_
from .loss import LrcDistillLoss
from .merge import merge_and_save, merged_state_dict
from .tuner import CoreLinear, CoreLRCAdapter, DenseLRCAdapter, PlainLRCAdapter

STAGE_PT = "pt"
STAGE_SFT = "sft"

__all__ = [
    "TwydLRCConfig",
    "build_core_basis",
    "check_basis",
    "build_outlier_aware_P",
    "ridge_zcol",
    "zero_perp_",
    "LrcDistillLoss",
    "merge_and_save",
    "merged_state_dict",
    "CoreLinear",
    "DenseLRCAdapter",
    "CoreLRCAdapter",
    "PlainLRCAdapter",
    "STAGE_PT",
    "STAGE_SFT",
]

