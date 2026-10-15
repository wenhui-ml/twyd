"""Synthetic tiny-model checks: no HF download, no GPU-train needed (part 1)."""
import torch

from twyd.config import TwydLRCConfig
from twyd.merge import merged_state_dict
from twyd.tuner import _build_factorized_student, _freeze_for_branch, _TwydWrapper, CoreLinear


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
        self.q_proj = torch.nn.Linear(hid, hid, bias=False)
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


def _wrap(branch, teacher, student, r):
    cfg = TwydLRCConfig(branch=branch, student_hidden=r,
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        kl_temperature=2.0, kl_weight=1.0, ntp_weight=1.0,
                        aux_weight=0.2, use_aux=True)
    _build_factorized_student(student, teacher, cfg, branch=branch)
    w = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(w, cfg)
    return w, cfg


def test_1_init_equivalence_dense_core_plain():
    H, d_ff, r = 24, 64, 12
    outs = {}
    for branch in ("dense", "core", "plain"):
        teacher, student = _make_pair(H, d_ff, r)
        w, _ = _wrap(branch, teacher, student, r)
        outs[branch] = {n: p.detach().clone() for n, p in merged_state_dict(w).items()}
    for k in outs["dense"]:
        assert torch.allclose(outs["dense"][k], outs["core"][k], atol=1e-5), k
        assert torch.allclose(outs["dense"][k], outs["plain"][k], atol=1e-5), k


def test_2_trainable_counts_reachable_set():
    H, d_ff, r, nlayer = 24, 64, 12, 2
    for branch in ("dense", "core", "plain"):
        teacher, student = _make_pair(H, d_ff, r)
        w, _ = _wrap(branch, teacher, student, r)
        mlp_params = [p for n, p in w.student.named_parameters()
                      if any(k in n for k in ("gate_proj", "up_proj", "down_proj")) and p.requires_grad]
        n = sum(p.numel() for p in mlp_params)
        if branch == "plain":
            assert n == nlayer * 3 * r * H, n
        else:
            assert n == nlayer * 3 * r * d_ff, (branch, n)
    teacher, student = _make_pair(H, d_ff, r)
    w, _ = _wrap("dense", teacher, student, r)
    attn = [p for n, p in w.student.named_parameters() if "q_proj" in n]
    assert attn and all(not p.requires_grad for p in attn)


def test_3_one_opt_step_moves_core_perp():
    torch.manual_seed(1)
    H, d_ff, r = 24, 64, 12
    teacher, student = _make_pair(H, d_ff, r)
    w, cfg = _wrap("core", teacher, student, r)
    zps = [m.z_perp for m in w.student.modules() if isinstance(m, CoreLinear)]
    assert zps and all(bool((zp == 0).all()) for zp in zps)
    opt = torch.optim.AdamW([p for p in w.student.parameters() if p.requires_grad], lr=1e-2)
    mods = [m for m in w.student.modules() if isinstance(m, CoreLinear)]
    before = [m.merged_weight().detach().clone() for m in mods]
    # gate/up take r-dim input, down takes d_ff-dim input
    x_r = torch.randn(4, r)
    x_ff = torch.randn(4, d_ff)
    loss = sum((m(x_ff if m.side == "right" else x_r) ** 2).mean() for m in mods)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(bool((zp != 0).any()) for zp in zps)
    after = [m.merged_weight().detach().clone()
             for m in w.student.modules() if isinstance(m, CoreLinear)]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_4_distillation_kl_aux_and_backward():
    from twyd.loss import LrcDistillLoss
    torch.manual_seed(2)
    H, d_ff, r = 24, 64, 12
    B, S, V = 2, 16, 64
    teacher, student = _make_pair(H, d_ff, r)
    w, cfg = _wrap("core", teacher, student, r)
    s_logits = torch.randn(B, S, V, requires_grad=True)
    t_logits = torch.randn(B, S, V)
    sh = torch.randn(B, S, r, requires_grad=True)
    th = torch.randn(B, S, H)

    class FakeTOut:
        def __init__(self):
            self.logits = t_logits
            self.hidden_states = (th,)

    w.get_teacher_outputs = lambda **kw: FakeTOut()
    w._twyd_last_inputs = {"input_ids": torch.randint(0, V, (B, S))}

    class FakeArgs:
        tuner_kwargs = {"kl_temperature": 2.0, "kl_weight": 1.0, "ntp_weight": 1.0,
                        "aux_weight": 0.2, "use_aux": True}

    class FakeTrainer:
        model = w
        _twyd_inputs = {}

    loss_fn = LrcDistillLoss(FakeArgs(), FakeTrainer())

    class FakeSOut:
        def __init__(self):
            self.logits = s_logits
            self.hidden_states = (sh,)

    labels = torch.randint(0, V, (B, S))
    val = loss_fn(FakeSOut(), labels)
    assert torch.isfinite(val) and float(val.detach()) > 0
    last = loss_fn.trainer._twyd_last
    assert last["kl"] is not None and last["kl"] > 0, last
    assert last["ntp"] is not None and last["ntp"] > 0, last
    assert last.get("aux", 0) > 0, last
    val.backward()
    assert s_logits.grad is not None and sh.grad is not None


def test_5_merge_roundtrip_deployable():
    H, d_ff, r = 24, 64, 12
    for branch in ("dense", "core", "plain"):
        teacher, student = _make_pair(H, d_ff, r)
        w, _ = _wrap(branch, teacher, student, r)
        sd = merged_state_dict(w)
        assert not any((".z_col" in k or ".z_perp" in k or "basis_" in k) for k in sd)
        for n, t in sd.items():
            if "gate_proj" in n or "up_proj" in n:
                assert tuple(t.shape) == (d_ff, r), (branch, n, tuple(t.shape))
            if "down_proj" in n:
                assert tuple(t.shape) == (r, d_ff), (branch, n, tuple(t.shape))


def test_6_plugin_registration():
    import twyd.plugin as P
    assert set(("dense_lrc", "core_lrc", "plain_lrc")) <= set(P._TUNERS)
    assert "lrc_distill" in P._LOSSES
    # swift base classes when installed; object fallback in bare env
    from twyd.tuner import DenseLRCAdapter
    names = [c.__name__ for c in type.mro(DenseLRCAdapter)]
    assert "DenseLRCAdapter" in names and ("Tuner" in names or "object" in names), names
    from twyd.loss import LrcDistillLoss
    names = [c.__name__ for c in type.mro(LrcDistillLoss)]
    assert "LrcDistillLoss" in names and ("BaseLoss" in names or "object" in names), names
    try:
        from swift.tuner_plugin.base import Tuner as _T
        from swift.loss.base import BaseLoss as _B
        assert issubclass(DenseLRCAdapter, _T)
        assert issubclass(LrcDistillLoss, _B)
    except ImportError:
        pass
