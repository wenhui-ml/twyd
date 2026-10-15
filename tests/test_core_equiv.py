"""Zero-init equivalence + merge equivalence (Prop.1) + loss smoke."""
import torch

from twyd.core_basis import build_core_basis, check_basis
from twyd.init import build_outlier_aware_P, ridge_zcol
from twyd.tuner import CoreLinear


def test_core_zero_init_matches_plain():
    torch.manual_seed(0)
    d_ff, H, r = 64, 24, 12
    W_teacher = torch.randn(d_ff, H)
    T, Up = build_core_basis(W_teacher, "left")
    qc = check_basis(T, Up)
    assert qc["ok"], qc
    # plain warm start
    W0 = torch.randn(d_ff, r)
    Z_col = (T.T @ W0).T.contiguous()
    plain = CoreLinear(T, Z_col, side="left", with_perp=False)
    core = CoreLinear(T, Z_col, Up, torch.zeros(r, d_ff - H), side="left", with_perp=True)
    x = torch.randn(2, r)
    assert torch.allclose(core(x), plain(x), atol=1e-5)
    # perp path is exactly zero at init
    assert torch.allclose(Up @ core.z_perp.T, torch.zeros_like(Up @ core.z_perp.T))


def test_merge_matches_forward():
    torch.manual_seed(1)
    d_ff, H, r = 48, 16, 8
    T, Up = build_core_basis(torch.randn(d_ff, H), "left")
    core = CoreLinear(T, torch.randn(r, H) * 0.02, Up,
                      torch.randn(r, d_ff - H) * 0.02, side="left", with_perp=True)
    x = torch.randn(3, r)
    W = core.merged_weight()
    assert W.shape == (d_ff, r)
    assert torch.allclose(x @ W.T, core(x), atol=1e-5)


def test_down_right_complement():
    torch.manual_seed(2)
    H, d_ff, r = 16, 48, 8
    from twyd.core_basis import build_down_basis
    R, Vp = build_down_basis(torch.randn(H, d_ff))
    core = CoreLinear(R, torch.randn(r, H) * 0.02, Vp,
                      torch.zeros(r, d_ff - H), side="right", with_perp=True)
    W = core.merged_weight()
    assert W.shape == (r, d_ff)
    # zero perp => col path only
    assert torch.allclose(W, core.z_col @ R, atol=1e-6)


def test_ridge_solve_recovers_prior_without_signal():
    torch.manual_seed(3)
    N, r, H = 32, 6, 4
    C = torch.randn(N, r)
    Wp = torch.randn(r, H)
    HT = C @ Wp  # no residual => solution == prior
    Z = ridge_zcol(C, HT, Wp, ridge_lambda=1e-3)
    assert torch.allclose(Z, Wp, atol=1e-4)


def test_lrc_loss_forward_smoke():
    from twyd.loss import LrcDistillLoss

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 40.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.2, "use_aux": True}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = None

    class FakeTrainer:
        model = FakeModel()
        _twyd_inputs = {}

    loss_fn = LrcDistillLoss(FakeArgs(), FakeTrainer())
    B, S, V = 1, 8, 64
    out = type("O", (), {"logits": torch.randn(B, S, V)})()
    labels = torch.randint(0, V, (B, S))
    val = loss_fn(out, labels)
    assert torch.isfinite(val) and float(val.detach()) > 0


def test_dense_branch_param_count_and_init():
    from twyd.tuner import _build_factorized_student
    from twyd.config import TwydLRCConfig
    torch.manual_seed(0)
    H, d_ff, r = 24, 64, 12

    class FakeMLP(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.gate_proj = torch.nn.Linear(in_f, out_f, bias=False)

    teacher = FakeMLP(H, d_ff)
    student = FakeMLP(r, d_ff)
    # fake configs
    teacher.config = type("C", (), {"hidden_size": H})()
    student.config = type("C", (), {"hidden_size": r})()
    cfg = TwydLRCConfig(branch="dense", student_hidden=r, target_modules=["gate_proj"])
    meta = _build_factorized_student(student, teacher, cfg, branch="dense")
    assert "gate_proj" in meta and meta["gate_proj"]["is_dense"]
    w = student.gate_proj.weight
    assert tuple(w.shape) == (d_ff, r), tuple(w.shape)  # full deployed matrix
    assert w.numel() == d_ff * r  # r*d_ff, not r*H


def test_outlier_aware_P_passthrough():
    torch.manual_seed(0)
    N, H, r = 64, 24, 12
    hidden = torch.randn(N, H) * 5
    P, idx = build_outlier_aware_P(hidden, r, outlier_keep=4)
    assert P.shape == (r, H) and len(idx) == 4
    for i, ch in enumerate(idx):
        assert P[i, ch] == 1.0


def test_loss_uses_cached_teacher_inputs():
    from twyd.loss import LrcDistillLoss

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 2.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.0, "use_aux": False}

    B, S, V = 1, 8, 32
    s_logits = torch.randn(B, S, V)
    t_logits = torch.randn(B, S, V)

    class FakeTeacherOut:
        def __init__(self, logits):
            self.logits = logits

    class FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.twyd_cfg = None
            self._twyd_last_inputs = {"input_ids": torch.randint(0, V, (B, S))}

        def get_teacher_outputs(self, **kw):
            return FakeTeacherOut(t_logits)

    class FakeTrainer:
        def __init__(self):
            self.model = FakeWrapper()
            self._twyd_inputs = {}

    loss_fn = LrcDistillLoss(FakeArgs(), FakeTrainer())
    out = type("O", (), {"logits": s_logits})()
    labels = torch.randint(0, V, (B, S))
    val = loss_fn(out, labels)
    assert torch.isfinite(val)
    assert loss_fn.trainer._twyd_last["kl"] is not None, "KL must flow via cached inputs"
