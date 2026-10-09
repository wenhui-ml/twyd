"""One-import registration for ms-swift: --external_plugins twyd.plugin."""
from .loss import LrcDistillLoss
from .tuner import CoreLRCAdapter, DenseLRCAdapter, PlainLRCAdapter

_TUNERS = {"dense_lrc": DenseLRCAdapter, "core_lrc": CoreLRCAdapter, "plain_lrc": PlainLRCAdapter}
_LOSSES = {"lrc_distill": LrcDistillLoss}

try:
    from swift.tuner_plugin.mapping import tuners_map as _tuners_map
    for _k, _v in _TUNERS.items():
        _tuners_map.setdefault(_k, _v)
except Exception:
    pass

try:
    from swift.loss.mapping import loss_map as _loss_map
    for _k, _v in _LOSSES.items():
        _loss_map.setdefault(_k, _v)
except Exception:
    pass

__all__ = ["_TUNERS", "_LOSSES"]
