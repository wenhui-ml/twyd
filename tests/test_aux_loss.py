"""Aux alignment must exist and carry gradients (regression test).

The original loss re-ran the student under ``torch.no_grad()`` and never asked
the teacher for hidden states, so ``TWYD_USE_AUX=true`` / lambda_aux=0.2 was a
silent no-op. This test locks the fixed path: teacher hidden_states requested,
student hidden_states in the graph, aux term non-None and differentiable.
"""
import types

import torch
from torch import nn

from twyd.config import TwydLRCConfig
from twyd.loss import LrcDistillLoss


class FakeStudent(nn.Module):
    def __init__(self, hidden=4, vocab=8):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.hidden = hidden
        self.vocab = vocab

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                output_hidden_states=False, use_cache=False):
        b, s = input_ids.shape
        h = torch.randn(b, s, self.hidden) * self.w
        logits = torch.randn(b, s, self.vocab) * self.w
        return types.SimpleNamespace(logits=logits, hidden_states=(h,))


class FakeWrapper(nn.Module):
    def __init__(self, hidden=4, vocab=8):
        super().__init__()
        self.twyd_cfg = TwydLRCConfig(use_aux=True, aux_weight=0.2,
                                      kl_temperature=2.0)
        self.student = FakeStudent(hidden, vocab)
        self._twyd_last_inputs = {}

    def get_teacher_outputs(self, output_hidden_states=False, **kw):
        ids = kw["input_ids"]
        b, s = ids.shape
        assert output_hidden_states, "teacher hidden states must be requested when use_aux"
        h = torch.randn(b, s, self.student.hidden)
        return types.SimpleNamespace(logits=torch.randn(b, s, self.student.vocab),
                                     hidden_states=(h,))


def test_aux_is_live_and_differentiable():
    torch.manual_seed(0)
    wrapper = FakeWrapper()
    ids = torch.randint(0, 8, (1, 6))

    class FakeTrainer:
        model = wrapper
        _twyd_inputs = {"input_ids": ids}

        def _set(self, d):
            self._twyd_last = d

    trainer = FakeTrainer()
    loss_fn = LrcDistillLoss(types.SimpleNamespace(), trainer)
    out = wrapper.student(input_ids=ids)
    loss = loss_fn(out, ids)
    assert torch.isfinite(loss)
    aux = trainer._twyd_last.get("aux")
    assert aux is not None and aux >= 0.0, f"aux was not computed: {trainer._twyd_last}"
    loss.backward()
    assert wrapper.student.w.grad is not None
    assert wrapper.student.w.grad.abs().sum() > 0
