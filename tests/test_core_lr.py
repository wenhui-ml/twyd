"""CORE-LR: freeze the plain-LRC col path, train a low-rank complement only."""
import json

import torch
import torch.nn.functional as F

from twyd.config import TwydLRCConfig, module_rank, parse_rank_map
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
        # num_hidden_layers is what layer-range parsing (`lora_layers`) reads.
        self.config = type("C", (), {"hidden_size": hid,
                                     "num_hidden_layers": nlayer})()


def _make_pair(H=24, d_ff=64, r=12):
    torch.manual_seed(0)
    teacher = FakeBlock(H, d_ff)
    student = FakeBlock(r, d_ff)
    with torch.no_grad():
        for m in teacher.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.1)
    return teacher, student


def _cfg(r, k, rank_map=""):
    return TwydLRCConfig(
        branch="core", student_hidden=r,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_rank=k, lora_alpha=16.0, lora_mode="core", freeze_col=True,
        rank_map=rank_map)


def _build(factorized=False, rank_map="", perp_basis="teacher"):
    H, d_ff, r, k = 24, 64, 12, 4
    teacher, student = _make_pair(H, d_ff, r)
    cfg = _cfg(r, k, rank_map=rank_map)
    cfg.factorized = factorized
    cfg.perp_basis = perp_basis
    meta = _build_factorized_student(student, teacher, cfg, branch="core")
    w = _TwydWrapper(student, teacher, cfg)
    w.twyd_meta = meta
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


def test_lora_b_gradient_is_nonzero_at_init():
    """B=0, A random: dL/dB must be nonzero or CORE-LR never starts learning."""
    w, mods, H, d_ff, r, k = _build()
    m = mods[0]
    x = torch.randn(2, r)
    m(x).sum().backward()
    assert m.lora_A.grad is not None and m.lora_B.grad is not None
    assert m.lora_B.grad.abs().sum() > 0
    assert m.lora_A.grad.abs().sum() == 0  # B=0 at step 0


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


def test_factorized_forward_matches_merged_and_gradients():
    """Cached-W_col + rank-k delta must be numerically equivalent to dense W."""
    w, mods, H, d_ff, r, k = _build(factorized=True)
    torch.manual_seed(7)
    for m in mods:
        expect = (d_ff, r) if m.side == "left" else (r, d_ff)
        assert m.factorized and m.W_col_cache.shape == expect
        with torch.no_grad():
            m.lora_A.normal_(std=0.05)
            m.lora_B.normal_(std=0.05)
        m.zero_grad(set_to_none=True)
        # gate/up consume the hidden width r; down consumes d_ff.
        in_dim = r if m.side == "left" else d_ff
        x = torch.randn(3, in_dim)
        up = torch.randn_like(m(x))
        y1 = m(x)
        (y1 * up).sum().backward()
        gA1, gB1 = m.lora_A.grad.clone(), m.lora_B.grad.clone()
        m.zero_grad(set_to_none=True)
        y2 = F.linear(x, m.merged_weight())
        (y2 * up).sum().backward()
        assert torch.allclose(y1, y2, atol=2e-5), (m.side, (y1 - y2).abs().max())
        assert torch.allclose(gA1, m.lora_A.grad, atol=2e-5, rtol=2e-4)
        assert torch.allclose(gB1, m.lora_B.grad, atol=2e-5, rtol=2e-4)


def test_factorized_offloads_basis_col_but_export_still_matches():
    """forward() parks basis_col on CPU; merged_weight() must still be exact."""
    w, mods, H, d_ff, r, k = _build(factorized=True)
    for m in mods:
        torch.manual_seed(3)
        with torch.no_grad():
            m.lora_A.normal_(std=0.05)
            m.lora_B.normal_(std=0.05)
        expect_before = m.merged_weight().clone()
        in_dim = r if m.side == "left" else d_ff
        x = torch.randn(2, in_dim)
        y = m(x)
        # After one forward the export-only basis is off the compute device...
        assert m.basis_col.device.type == "cpu"
        # ...but export (used by merge/convert) is still bit-comparable.
        assert torch.allclose(m.merged_weight(), expect_before, atol=1e-6)
        # And the deployed weight agrees with the factorized forward.
        y2 = F.linear(x, m.merged_weight())
        assert torch.allclose(y, y2, atol=2e-5)
        # refresh_cache() (used after an external z_col write) still works.
        m.refresh_cache()
        y3 = m(x)
        assert torch.allclose(y3, y2, atol=2e-5)


def test_rank_map_resolves_per_module_and_builds_variable_k():
    spec = json.dumps({"layers.0.gate_proj": 2, "down_proj": 3})
    rm = parse_rank_map(spec)
    assert module_rank(rm, "model.layers.0.gate_proj", 4) == 2
    assert module_rank(rm, "model.layers.1.down_proj", 4) == 3
    assert module_rank(rm, "model.layers.1.up_proj", 4) == 4
    w, mods, H, d_ff, r, k = _build(rank_map=spec)
    ranks = {n: v["lora_rank"] for n, v in w.twyd_meta.items()}
    assert ranks["layers.0.gate_proj"] == 2
    assert ranks["layers.1.down_proj"] == 3
    assert ranks["layers.1.up_proj"] == 4
    # bare "down_proj" applies to every layer: L0 gate2/up4/down3,
    # L1 gate4/up4/down3.
    expected = (2 + 4 + 3 + 4 + 4 + 3) * (r + d_ff - H)
    assert sum(p.numel() for p in w.student.parameters() if p.requires_grad) == expected


def test_dense_projection_selection_trains_only_selected_projections():
    """Projection-level budget: only the named MLP projections stay trainable."""
    H, d_ff, r = 24, 64, 12
    teacher, student = _make_pair(H, d_ff, r)
    cfg = TwydLRCConfig(branch="dense", student_hidden=r,
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        lora_rank=0, lora_modules="gate_proj")
    meta = _build_factorized_student(student, teacher, cfg, branch="dense")
    w = _TwydWrapper(student, teacher, cfg)
    w.twyd_cfg = cfg
    w.twyd_meta = meta
    _freeze_for_branch(w, cfg)

    names = [n for n, p in w.student.named_parameters() if p.requires_grad]
    assert names and all(".gate_proj." in n for n in names), names
    # 2 layers x one projection x (d_ff x r)
    assert sum(p.numel() for p in w.student.parameters() if p.requires_grad) == 2 * r * d_ff


def test_dense_layer_selection_trains_only_selected_layers():
    """Layer-selective full-MLP (paper Fig.8b): only `lora_layers` stay trainable.

    On two consumer GPUs the memory knob is *which layers* get the full-MLP
    weight, not a rank-k factorization: unfreezing 6 of 28 layers cuts the
    trainable/optimizer state by ~5x with the same deployed shape.
    """
    H, d_ff, r = 24, 64, 12
    teacher, student = _make_pair(H, d_ff, r)
    cfg = TwydLRCConfig(branch="dense", student_hidden=r,
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        lora_rank=0, lora_layers="0-0")
    meta = _build_factorized_student(student, teacher, cfg, branch="dense")
    assert meta, "builder produced no modules"
    w = _TwydWrapper(student, teacher, cfg)
    w.twyd_cfg = cfg
    w.twyd_meta = meta
    _freeze_for_branch(w, cfg)

    names = [n for n, p in w.student.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("layers.0.") for n in names), names
    # 3 MLP projections of a d_ff x r dense weight, one layer only
    n_train = sum(p.numel() for p in w.student.parameters() if p.requires_grad)
    assert n_train == 3 * r * d_ff, n_train


def test_col_completion_control_same_params_but_no_new_directions():
    """P6 (paper §6.1): equal trainable count, reachable set stays in col(T)."""
    from twyd.core_basis import build_core_basis

    w_col, _mods_col, H, d_ff, r, k = _build(perp_basis="col")
    w_t, _mods_t, *_ = _build(perp_basis="teacher")
    n_col = sum(p.numel() for p in w_col.student.parameters() if p.requires_grad)
    n_t = sum(p.numel() for p in w_t.student.parameters() if p.requires_grad)
    assert n_col == n_t == 2 * 3 * k * (r + (d_ff - H))

    named_col = {n: m for n, m in w_col.student.named_modules()
                 if isinstance(m, CoreLinearLoRA)}
    named_t = {n: m for n, m in w_t.student.named_modules()
               if isinstance(m, CoreLinearLoRA)}
    assert named_col and named_col.keys() == named_t.keys()
    tmods = dict(w_col.teacher.named_modules())

    for name, m in named_col.items():
        mt = named_t[name]
        # Same shapes => identical parameter budget as CORE-LR.
        assert m.basis_perp.shape == mt.basis_perp.shape
        with torch.no_grad():
            m.lora_A.normal_(std=0.05)
            m.lora_B.normal_(std=0.05)
            mt.lora_A.copy_(m.lora_A)
            mt.lora_B.copy_(m.lora_B)
        tw = tmods[name].weight.detach().float()
        W, Wt = m.merged_weight().float(), mt.merged_weight().float()
        if m.side == "left":
            T, U = build_core_basis(tw, "left")
            dW, dWt = W - T @ (T.T @ W), Wt - T @ (T.T @ Wt)
            assert (U.T @ dW).abs().max() < 1e-4      # nothing outside col(T)
            assert (U.T @ dWt).abs().max() > 1e-3     # teacher basis opens it
        else:
            R, V = build_core_basis(tw, "right")
            dW, dWt = W - (W @ R.T) @ R, Wt - (Wt @ R.T) @ R
            assert (dW @ V.T).abs().max() < 1e-4
            assert (dWt @ V.T).abs().max() > 1e-3
