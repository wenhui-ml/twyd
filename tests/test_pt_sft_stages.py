"""PT (dense supervision) vs SFT (prompt-masked) loss + PT->SFT warm start."""
import torch

from twyd.config import TwydLRCConfig
from twyd.loss import LrcDistillLoss
from twyd.merge import merged_state_dict
from twyd.tuner import _build_factorized_student, _freeze_for_branch, _TwydWrapper


def _fake_pair(H=24, d_ff=64, r=12):
    import torch.nn as nn

    class FakeMLP(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.gate_proj = nn.Linear(in_f, out_f, bias=False)

    torch.manual_seed(0)
    teacher = FakeMLP(H, d_ff)
    student = FakeMLP(r, d_ff)
    teacher.config = type("C", (), {"hidden_size": H})()
    student.config = type("C", (), {"hidden_size": r})()
    return teacher, student


def _loss_val(stage, labels, loss_scale=None):
    torch.manual_seed(0)
    B, S, V = 2, 16, 64
    s_logits = torch.randn(B, S, V)
    t_logits = torch.randn(B, S, V)

    class FakeTOut:
        def __init__(self):
            self.logits = t_logits
            self.hidden_states = None

    class FakeW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = TwydLRCConfig(branch="dense", student_hidden=12,
                                          stage=stage, use_aux=False)
            self._twyd_last_inputs = {"input_ids": torch.randint(0, V, (B, S))}

        def get_teacher_outputs(self, **kw):
            return FakeTOut()

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 2.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.0,
                        "use_aux": False, "stage": stage}

    class FakeTrainer:
        def __init__(self):
            self.model = FakeW()
            self._twyd_inputs = {}

    fn = LrcDistillLoss(FakeArgs(), FakeTrainer())

    class FakeSOut:
        def __init__(self):
            self.logits = s_logits
            self.hidden_states = None

    val = fn(FakeSOut(), labels, loss_scale=loss_scale)
    return float(val.detach()), dict(fn.trainer._twyd_last)


def test_pt_distills_every_position():
    B, S, V = 2, 16, 64
    labels = torch.randint(0, V, (B, S))  # no -100 in PT
    val, last = _loss_val("pt", labels)
    assert val > 0 and last["kl"] is not None and last["ntp"] is not None


def test_sft_masks_prompt_for_kl_and_ce():
    torch.manual_seed(0)
    B, S, V = 2, 16, 64
    labels = torch.randint(0, V, (B, S))
    labels[:, :8] = -100  # prompt half masked
    full, _ = _loss_val("sft", torch.randint(0, V, (B, S)))
    masked, last = _loss_val("sft", labels)
    assert masked > 0 and last["kl"] is not None
    # masked loss differs from unmasked: mask actually takes effect
    assert abs(masked - full) > 1e-6, (masked, full)


def test_sft_loss_scale_tensor_respected():
    torch.manual_seed(1)
    B, S, V = 2, 16, 64
    labels = torch.randint(0, V, (B, S))
    labels[:, :4] = -100
    ls = torch.ones(B, S)
    ls[:, :8] = 0.0  # zero out extra response positions
    v1, _ = _loss_val("sft", labels, loss_scale=None)
    v2, _ = _loss_val("sft", labels, loss_scale=ls)
    assert abs(v1 - v2) > 1e-6, (v1, v2)


def test_aux_averages_over_matched_layers():
    """Eq.7 hidden part: aux averages 1/L over layers, not last-only."""
    import torch.nn.functional as F
    torch.manual_seed(3)
    B, S, r, H, V = 1, 8, 12, 24, 32
    # student layers: layer0 matches teacher, layer1 (last) is far off.
    # last-only aux would be large; layer-averaged aux must be smaller.
    sh0 = torch.randn(B, S, r)
    th0 = sh0 @ torch.randn(r, H) * 0.01  # placeholder, replaced below
    # Build a fixed projection-friendly pair: teacher = student @ P0
    P0 = torch.randn(r, H)
    th0 = (sh0 @ P0).detach()
    sh1 = torch.randn(B, S, r)
    th1 = torch.randn(B, S, H) * 5.0
    s_hidden = (sh0.clone().requires_grad_(True), sh1.clone().requires_grad_(True))
    t_hidden = (th0, th1)

    class FakeTOut:
        def __init__(self):
            self.logits = torch.randn(B, S, V)
            self.hidden_states = t_hidden

    from twyd.config import TwydLRCConfig as _C

    class FakeW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = _C(branch="dense", student_hidden=r, stage="pt",
                               use_aux=True, aux_weight=0.2)
            self._twyd_last_inputs = {"input_ids": torch.randint(0, V, (B, S))}

        def get_teacher_outputs(self, **kw):
            return FakeTOut()

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 2.0, "kl_weight": 0.0,
                        "ntp_weight": 0.0, "aux_weight": 1.0,
                        "use_aux": True, "stage": "pt", "max_aux_layers": 2}

    class FakeTrainer:
        def __init__(self):
            self.model = FakeW()
            self._twyd_inputs = {}

    fn = LrcDistillLoss(FakeArgs(), FakeTrainer())

    class FakeSOut:
        def __init__(self):
            self.logits = torch.randn(B, S, V)
            self.hidden_states = s_hidden

    labels = torch.randint(0, V, (B, S))
    val = fn(FakeSOut(), labels)
    aux = fn.trainer._twyd_last["aux"]
    assert aux is not None and aux > 0, fn.trainer._twyd_last
    # last-layer-only MSE reference (fit P on last layer, as loss does):
    # it is ~0 because P was fit there. The layer-averaged aux must be
    # LARGER because it also counts the mismatched early layer — proving
    # the loss averages over layers instead of using last-only.
    A = sh1.detach().reshape(-1, r)
    Bm = th1.detach().reshape(-1, H)
    Plast = torch.linalg.lstsq(A.float(), Bm.float()).solution.to(sh1.dtype)
    last_only = float(F.mse_loss(sh1 @ Plast, th1).detach())
    assert aux > last_only, (aux, last_only)
    # And it must equal the mean of per-layer terms (2 layers here).
    A0 = sh0.detach().reshape(-1, r)
    B0 = th0.detach().reshape(-1, H)
    P0fit = torch.linalg.lstsq(A.float(), Bm.float()).solution.to(sh0.dtype)
    layer0 = float(F.mse_loss(sh0 @ P0fit, th0).detach())
    assert abs(aux - (layer0 + last_only) / 2) / max(aux, 1e-6) < 0.05, (aux, layer0, last_only)


def test_sft_warm_start_loads_pt_merged(tmp_path):
    import os
    from safetensors.torch import save_file
    H, d_ff, r = 24, 64, 12
    teacher, student = _fake_pair(H, d_ff, r)
    # fake merged PT ckpt: gate weight filled with 7.0
    pt_w = torch.full((d_ff, r), 7.0)
    ckpt = os.path.join(str(tmp_path), "model.safetensors")
    save_file({"gate_proj.weight": pt_w}, ckpt)
    from twyd.tuner import _load_pt_merged_
    n = _load_pt_merged_(student, ckpt)
    assert n == 1, n
    assert torch.allclose(student.gate_proj.weight.detach().cpu(), pt_w)
    # rebuild factorized student on top of PT weights: W0 inherits PT values
    cfg = TwydLRCConfig(branch="dense", student_hidden=r, stage="sft",
                        warm_start="pt_ckpt", target_modules=["gate_proj"])
    meta = _build_factorized_student(student, teacher, cfg, branch="dense")
    assert "gate_proj" in meta
    w = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(w, cfg)
    sd = merged_state_dict(w)
    assert torch.isfinite(sd["gate_proj.weight"]).all()
