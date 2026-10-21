"""The teacher must be moved to the compute device ONCE, not once per step.

The teacher is loaded on CPU on purpose (VRAM budget), so its weights have to
reach the GPU before each forward.  The original code rebuilt
``{name: p.to(device)}`` on every call, i.e. re-uploaded the full 3.2B-param
Llama-3.2-3B teacher (6.4 GB bf16) per training step.  On a 7900 XTX that was
1.4-1.9 s/step with the GPU idle >90% of the time (``TWYD_PROFILE=1`` showed
``teacher=1431ms`` of a ``loss_total=1444ms``).  These tests pin the cache.
"""
import os

import torch

from twyd.config import TwydLRCConfig
from twyd.tuner import _TwydWrapper


def _pair():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            self.register_buffer("scale", torch.ones(1))

    torch.manual_seed(0)
    student, teacher = Tiny(), Tiny()
    cfg = TwydLRCConfig(branch="core", student_hidden=4)
    return _TwydWrapper(student, teacher, cfg), teacher


def test_teacher_params_moved_once_and_reused():
    w, _ = _pair()
    dev = torch.device("cpu")
    p1, b1 = w._teacher_dev_params(dev, _fcall=object())
    p2, b2 = w._teacher_dev_params(dev, _fcall=object())
    # Same cached objects => no repeated host->device copy on later steps.
    assert p1 is p2 and b1 is b2
    assert set(p1) == {"lin.weight"} and "scale" in b1


def test_teacher_device_cpu_opt_out_skips_cache():
    w, _ = _pair()
    old = os.environ.get("TWYD_TEACHER_DEVICE")
    os.environ["TWYD_TEACHER_DEVICE"] = "cpu"
    try:
        p1, _ = w._teacher_dev_params(torch.device("cpu"), _fcall=object())
        p2, _ = w._teacher_dev_params(torch.device("cpu"), _fcall=object())
        # Opt-out path builds fresh dicts each call (old behaviour) and never
        # populates the cache.
        assert p1 is not p2
        assert w._teacher_dev_cache.get((str(torch.device("cpu")), True)) is None
    finally:
        if old is None:
            del os.environ["TWYD_TEACHER_DEVICE"]
        else:
            os.environ["TWYD_TEACHER_DEVICE"] = old


def test_teacher_cache_survives_forward_calls():
    """Two teacher forwards in a row must not re-copy parameters."""
    w, teacher = _pair()
    calls = []
    real_to = torch.Tensor.to

    def counting_to(self, *a, **kw):
        calls.append(1)
        return real_to(self, *a, **kw)

    ids = torch.zeros(1, 3, dtype=torch.long)
    w._teacher_dev_params(torch.device("cpu"), _fcall=object())
    n_after_first = len(calls)
    w._teacher_dev_params(torch.device("cpu"), _fcall=object())
    assert len(calls) == n_after_first, "second call re-copied teacher weights"
    assert ids.dtype == torch.long