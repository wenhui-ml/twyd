"""One-import registration for ms-swift: --external_plugins twyd.plugin."""
from .loss import LrcDistillLoss
from .moe import MoeLRCAdapter
from .tuner import CoreLRCAdapter, DenseLRCAdapter, PlainLRCAdapter

try:
    from transformers import TrainerCallback as _Cb

    class TwydInputCacheCallback(_Cb):
        """Cache raw batch inputs on trainer._twyd_inputs before swift strips them.

        ms-swift seq2seq_trainer.compute_loss pops labels/compute_loss_func and
        template.compute_sft_loss consumes inputs; loss_fn then cannot see
        input_ids for the teacher forward. This callback runs first and saves
        the tensors loss needs. Register via --callbacks twyd.plugin.TwydInputCacheCallback
        or rely on wrapper._twyd_last_inputs fallback.
        """

        def on_step_begin(self, args, state, control, **kwargs):
            inputs = kwargs.get("inputs", None)
            trainer = kwargs.get("trainer", None)
            if inputs is not None and trainer is not None:
                try:
                    keep = {k: v for k, v in inputs.items()
                            if k in ("input_ids", "attention_mask", "position_ids")}
                    if keep:
                        trainer._twyd_inputs = keep
                except Exception:
                    pass

        # transformers Trainer calls training_step with inputs positionally;
        # on_step_begin has no inputs param, so also hook on_train_batch_start
        # variants where available. Wrapper cache remains the primary path.
        def on_train_batch_start(self, args, state, control, **kwargs):
            return self.on_step_begin(args, state, control, **kwargs)
except Exception:  # transformers not available at import time
    TwydInputCacheCallback = None  # type: ignore

_TUNERS = {
    "dense_lrc": DenseLRCAdapter,
    "core_lrc": CoreLRCAdapter,
    "plain_lrc": PlainLRCAdapter,
    "moe_lrc": MoeLRCAdapter,
}
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

__all__ = ["_TUNERS", "_LOSSES", "TwydInputCacheCallback"]
