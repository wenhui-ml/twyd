"""CORE-LR: freeze the plain-LRC col path, train a low-rank complement only."""
import torch

from twyd.config import TwydLRCConfig
from twyd.tuner import (CoreLinearLoRA, _build_factorized_student,
                        _freeze_for_branch, _TwydWrapper)


class FakeMLP(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.gate_proj = torch.nn.Linear(in_f, out_f, bias=False)
        self.up_proj = torch.nn.Linear(in_f, out_f, bias=False)
        self.down_proj = torch.nn.Linear(out_f, in_f, bias=False)


class FakeBlock(torch.nn.Module):
    def __init__(self, hid, ff, nlayer=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeMLP(hid, ff) for _ in range(nlayer)])
        self.config = type("C", (), {"hidden_size": hid})()


def _make_pair(H=24, d_ff=64, r=12):
    torch.manual_seed(0)
    teacher = FakeBlock(H, d_ff)
    student = FakeBlock(r, d_ff)
    with torch.no_grad():
        for m in teacher.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.1)
    return teacher, student


def _cfg(r, k):
    return TwydLRCConfig(
        branch="core", student_hidden=r,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_rank=k, lora_alpha=16.0, lora_mode="core", freeze_col=True)


def _build():
    H, d_ff, r, k = 24, 64, 12, 4
    teacher, student = _make_pair(H, d_ff, r)
    cfg = _cfg(r, k)
    _build_factorized_student(student, teacher, cfg, branch="core")
    w = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(w, cfg)
    mods = [m for m in w.student.modules() if isinstance(m, CoreLinearLoRA)]
    assert mods
    return w, mods, H, d_ff, r, k


def test_col_is_frozen_and_only_complement_trains():
    w, mods, H, d_ff, r, k = _build()
    for m in mods:
        assert not m.z_col.requires_grad
        assert m.lora_A.requires_grad and m.lora_B.requires_grad
    trainable = [n for n, p in w.student.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n.lower() for n in trainable), trainable[:5]
    # k*(r + (d_ff-H)) per projection, 3 projections, 2 layers
    expected = 2 * 3 * k * (r + (d_ff - H))
    assert sum(p.numel() for p in w.student.parameters() if p.requires_grad) == expected


def test_init_equals_plain_and_delta_lives_in_complement():
    w, mods, H, d_ff, r, k = _build()
    with torch.no_grad():
        for m in mods:
            if m.side == "left":
                wcol = m.basis_col @ m.z_col.T
            else:
                wcol = m.z_col @ m.basis_col
            assert torch.allclose(m.merged_weight(), wcol, atol=1e-6)
            m.lora_A.normal_(std=0.1)
            m.lora_B.normal_(std=0.1)
    for m in mods:
        if m.side == "left":
            wcol = m.basis_col @ m.z_col.T
        else:
            wcol = m.z_col @ m.basis_col
        delta = m.merged_weight() - wcol
        if m.side == "left":
            zp = m.basis_perp.T @ delta
            assert torch.allclose(m.basis_perp @ zp, delta, atol=1e-5)
        else:
            zp = delta @ m.basis_perp.T
            assert torch.allclose(zp @ m.basis_perp, delta, atol=1e-5)
