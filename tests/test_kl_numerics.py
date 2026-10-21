"""Regression: distillation KL must not be computed in bf16.

A 1.5B CORE-LR run on the official LRC-1.5B-Base printed

    {'loss': '-3.348', ...}  ->  {'train_loss': '-3.394', ...}

with a healthy token_acc.  Root cause: with T=40 the logits are divided down to
~1e-2, while bf16 resolution near ``log p ~ -log(V) ~ -11.8`` (V=128k) is
~5e-2, so the log-probs were pure rounding noise; multiplying by ``T*T`` only
blew that noise up.  These tests pin the fp32 path.
"""
import torch
import torch.nn.functional as F

from twyd.config import TwydLRCConfig
from twyd.loss import LrcDistillLoss


def _loss_fn():
    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 40.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.0,
                        "use_aux": False, "stage": "sft"}

    class FakeW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = TwydLRCConfig(branch="dense", student_hidden=12,
                                          stage="sft", use_aux=False)

    class FakeTrainer:
        def __init__(self):
            self.model = FakeW()
            self._twyd_inputs = {}

    return LrcDistillLoss(FakeArgs(), FakeTrainer())


def _fp32_reference(shift_s, shift_t, T):
    logp_s = F.log_softmax(shift_s.float() / T, dim=-1)
    logp_t = F.log_softmax(shift_t.float() / T, dim=-1)
    return (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1) * (T * T)


def test_per_pos_kl_non_negative_in_bf16_inputs():
    torch.manual_seed(0)
    B, S, V, T = 2, 12, 4096, 40.0
    # Realistic magnitude for a 1.5B LM's logits.
    t = (torch.randn(B, S, V) * 8.0).to(torch.bfloat16)
    s = (t.float() + torch.randn(B, S, V) * 0.5).to(torch.bfloat16)
    kl = _loss_fn()._per_pos_kl(s, t, T)
    assert kl.dtype == torch.float32
    assert torch.isfinite(kl).all()
    assert (kl >= 0).all(), f"negative per-position KL: min={kl.min().item()}"


def test_bf16_naive_kl_is_numerically_broken():
    """Locks in WHY the fp32 cast exists (guards against a silent revert)."""
    torch.manual_seed(0)
    B, S, V, T = 1, 8, 8192, 40.0
    t = (torch.randn(B, S, V) * 8.0).to(torch.bfloat16)
    s = t.clone()
    # Naive old path: log_softmax in bf16, then T*T rescaling.
    logp_s = F.log_softmax(s / T, dim=-1)
    prob_t = F.softmax(t / T, dim=-1)
    naive = F.kl_div(logp_s, prob_t, reduction="none").sum(dim=-1) * (T * T)
    fixed = _loss_fn()._per_pos_kl(s, t, T)
    # Identical distributions => KL must be 0.  The bf16 path returns garbage.
    assert fixed.abs().max() < 1e-3
    assert naive.abs().max() > 10 * max(fixed.abs().max(), 1e-6) or \
        naive.min() < -1e-3, f"naive bf16 KL unexpectedly clean: {naive.abs().max()}"


def test_per_pos_kl_matches_fp32_reference_and_chunking():
    torch.manual_seed(0)
    B, S, V, T = 1, 40, 512, 40.0
    t = torch.randn(B, S, V) * 6.0
    s = t + torch.randn(B, S, V) * 2.0
    fn = _loss_fn()
    ref = _fp32_reference(s, t, T)
    got = fn._per_pos_kl(s, t, T)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-3)
    # Chunking must be value-identical (and stay differentiable).
    chunked = fn._per_pos_kl(s, t, T, chunk=7)
    assert torch.allclose(chunked, got, rtol=1e-5, atol=1e-5)
    s_g = s.clone().requires_grad_(True)
    fn._per_pos_kl(s_g, t, T, chunk=7).sum().backward()
    assert s_g.grad is not None and torch.isfinite(s_g.grad).all()
    assert s_g.grad.abs().sum() > 0


def test_sft_loss_is_non_negative_end_to_end():
    """Total SFT loss (CE + KL, aux off) must never go negative."""
    torch.manual_seed(0)
    B, S, V = 1, 16, 2048
    t_logits = torch.randn(B, S, V) * 8.0
    s_logits = (t_logits + torch.randn(B, S, V)).to(torch.bfloat16)

    class FakeTOut:
        logits = t_logits
        hidden_states = None

    class FakeW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = TwydLRCConfig(branch="dense", student_hidden=12,
                                          stage="sft", use_aux=False)
            self._twyd_last_inputs = {"input_ids": torch.randint(0, V, (B, S))}

        def get_teacher_outputs(self, **kw):
            return FakeTOut()

    class FakeTrainer:
        def __init__(self):
            self.model = FakeW()
            self._twyd_inputs = {}

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 40.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.0,
                        "use_aux": False, "stage": "sft"}

    class FakeSOut:
        logits = s_logits
        hidden_states = None

    fn = LrcDistillLoss(FakeArgs(), FakeTrainer())
    labels = torch.randint(0, V, (B, S))
    labels[:, :6] = -100  # prompt masked
    loss = fn(FakeSOut(), labels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0, f"negative SFT loss: {loss.item()}"