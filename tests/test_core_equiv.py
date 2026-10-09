"""Zero-init equivalence + merge equivalence (Prop.1) + loss smoke."""
import torch

from twyd.core_basis import build_core_basis, check_basis
from twyd.init import ridge_zcol
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
    assert torch.isfinite(val) and float(val) > 0
