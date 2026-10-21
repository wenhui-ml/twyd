"""Data-aware ridge warm start (paper Alg.1): math + tiny end-to-end checks."""
import json

import pytest
import torch
from torch import nn

from twyd.config import TwydLRCConfig
from twyd.core_basis import build_core_basis
from twyd.init import (build_master_P, calibrate_attention, calibrate_ridge,
                       collect_mlp_groups, factorize_deployed,
                       head_projection_mats, load_calib_ids, pack_calib_ids,
                       project_student_with_P_, ridge_deployed)
from twyd.tuner import (CoreLinear, CoreLinearLoRA, _build_factorized_student,
                        _shrink_student_)


class TinyMLP(nn.Module):
    def __init__(self, hid, ff):
        super().__init__()
        self.gate_proj = nn.Linear(hid, ff, bias=False)
        self.up_proj = nn.Linear(hid, ff, bias=False)
        self.down_proj = nn.Linear(ff, hid, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, hid, ff):
        super().__init__()
        self.mlp = TinyMLP(hid, ff)

    def forward(self, x):
        return x + self.mlp(x)


class TinyLM(nn.Module):
    def __init__(self, vocab, hid, ff, nlayer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hid)
        self.layers = nn.ModuleList([TinyBlock(hid, ff) for _ in range(nlayer)])
        self.config = type("C", (), {"hidden_size": hid, "num_hidden_layers": nlayer})()

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


def _pair(H=32, d_ff=96, r=16, V=128, nlayer=2):
    torch.manual_seed(0)
    teacher = TinyLM(V, H, d_ff, nlayer)
    student = TinyLM(V, r, d_ff, nlayer)
    with torch.no_grad():
        for p in teacher.parameters():
            p.normal_(std=0.1)
        student.embed.weight.copy_(teacher.embed.weight[:, :r])
        for p in student.parameters():
            if p is not student.embed.weight:
                p.normal_(std=0.1)
    return teacher, student


def _cfg(r, branch="core", **kw):
    return TwydLRCConfig(branch=branch, student_hidden=r,
                         target_modules=["gate_proj", "up_proj", "down_proj"],
                         warm_start="ridge", **kw)


def _capture(model, ids, specs):
    from twyd import init as I
    return I._forward_capture(model, {"input_ids": ids}, specs)


def _gate_mse(student, teacher, ids):
    groups = collect_mlp_groups(student, teacher, ["gate_proj", "up_proj", "down_proj"])
    specs_s, specs_t = {}, {}
    for gi, g in enumerate(groups):
        for short in ("gate_proj", "up_proj"):
            if short in g["projs"]:
                s, t = g["projs"][short]
                specs_s[f"{gi}_{short}"] = (s, "post")
                specs_t[f"{gi}_{short}"] = (t, "post")
    ss = _capture(student, ids, specs_s)
    tt = _capture(teacher, ids, specs_t)
    vals = []
    for k in ss:
        s, t = ss[k], tt[k]
        n = min(s.shape[0], t.shape[0])
        vals.append(torch.mean((s[:n] - t[:n]) ** 2).item())
    return sum(vals) / max(len(vals), 1)


def test_ridge_deployed_and_factorize_roundtrip():
    torch.manual_seed(1)
    N, in_f, out_f = 256, 16, 96
    C = torch.randn(N, in_f)
    W = torch.randn(out_f, in_f)
    HT = C @ W.T
    W_hat, st = ridge_deployed(C, HT, W_prior=torch.zeros_like(W), ridge_lambda=1e-6)
    assert (W_hat - W).abs().max() < 1e-4
    assert st["rel_err_after"] < 1e-2
    # left complement split is exact
    Wt = torch.randn(out_f, 32)
    T, U = build_core_basis(Wt, "left")
    Zc = torch.randn(in_f, 32)
    Zp = torch.randn(in_f, out_f - 32)
    Wfull = T @ Zc.T + U @ Zp.T
    Zc2, Zp2, _ = factorize_deployed(Wfull, Wt, "left")
    assert torch.allclose(T @ Zc2.T + U @ Zp2.T, Wfull, atol=1e-5)
    # right split is exact
    Wt2 = torch.randn(16, out_f)
    R, V = build_core_basis(Wt2, "right")
    Zc = torch.randn(in_f, 16)
    Zp = torch.randn(in_f, out_f - 16)
    Wfull2 = Zc @ R + Zp @ V
    Zc3, Zp3, _ = factorize_deployed(Wfull2, Wt2, "right")
    assert torch.allclose(Zc3 @ R + Zp3 @ V, Wfull2, atol=1e-5)


def test_master_p_slice_matches_shrink():
    """P = first r rows must reproduce the existing slice warm start exactly."""
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    teacher = AutoModelForCausalLM.from_config(cfg).eval()
    s1 = _shrink_student_(teacher, teacher, 16)
    s2 = _shrink_student_(teacher, teacher, 16)
    P = torch.eye(32)[:16]
    project_student_with_P_(s2, teacher, P)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        mod = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        w1 = s1.get_submodule(f"model.layers.0.{mod}.{name}").weight
        w2 = s2.get_submodule(f"model.layers.0.{mod}.{name}").weight
        assert torch.allclose(w1, w2, atol=1e-3), name
    assert torch.allclose(s1.get_input_embeddings().weight,
                          s2.get_input_embeddings().weight, atol=1e-3)


def test_shrink_preserves_attn_impl():
    """Width-shrink must not silently drop --attn_impl (e.g. flash_attention_2)."""
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    teacher = AutoModelForCausalLM.from_config(cfg).eval()
    teacher.config._attn_implementation = "sdpa"
    shrunk = _shrink_student_(teacher, teacher, 16)
    assert shrunk.config._attn_implementation == "sdpa"


def test_master_p_projection_tiny_hf(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    teacher = AutoModelForCausalLM.from_config(cfg).eval()
    student = _shrink_student_(teacher, teacher, 16)
    P = build_master_P(teacher, 16)
    assert P.shape == (16, 32)
    assert torch.allclose(P @ P.T, torch.eye(16), atol=1e-5)
    n = project_student_with_P_(student, teacher, P)
    assert n > 0
    E_t = teacher.get_input_embeddings().weight.detach()
    E_s = student.get_input_embeddings().weight.detach()
    assert torch.allclose(E_s, E_t @ P.T, atol=1e-5)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        mod = student.get_submodule(f"model.layers.0.self_attn.{name}") \
            if name.endswith("_proj") and name in ("q_proj", "k_proj", "v_proj", "o_proj") \
            else student.get_submodule(f"model.layers.0.mlp.{name}")
        assert torch.isfinite(mod.weight).all(), name
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out = student(input_ids=ids)
    assert out.logits.shape == (2, 16, 128)


def test_calibrate_attention_tiny_hf():
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    teacher = AutoModelForCausalLM.from_config(cfg).eval()
    student = _shrink_student_(teacher, teacher, 16)
    P = torch.eye(32)[:16]
    S_q, S_kv = head_projection_mats(cfg, 16, P, "select")
    # LRC keeps teacher heads: q/k/v width = 4*8 = 32, kv width = 2*8 = 16.
    assert S_q.shape == (32, 32) and S_kv.shape == (16, 16)
    S_q2, S_kv2 = head_projection_mats(cfg, 16, P, "merge")
    assert torch.allclose(S_q2, torch.eye(32), atol=1e-6)
    assert torch.allclose(S_kv2, torch.eye(16), atol=1e-6)
    q_before = student.get_submodule("model.layers.0.self_attn.q_proj").weight.detach().clone()
    ids = torch.randint(0, 128, (4, 16))
    rep = calibrate_attention(student, teacher, ids, P=P, head_mode="select",
                              batch_size=2, log=None)
    assert rep["n_groups"] == 2
    q_after = student.get_submodule("model.layers.0.self_attn.q_proj").weight.detach()
    assert not torch.allclose(q_before, q_after)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = student.get_submodule(f"model.layers.0.self_attn.{n}").weight
        assert torch.isfinite(w).all(), n
    with torch.no_grad():
        out = student(input_ids=ids)
    assert out.logits.shape == (4, 16, 128)


def test_calibrate_ridge_improves_mlp_and_sets_residual():
    H, d_ff, r, V = 32, 96, 16, 128
    teacher, student = _pair(H, d_ff, r, V)
    cfg = _cfg(r, ridge_perp="residual")
    _build_factorized_student(student, teacher, cfg, branch="core")
    ids = torch.randint(0, V, (8, 32))
    before = _gate_mse(student, teacher, ids)
    rep = calibrate_ridge(student, teacher, ids, target_modules=cfg.target_modules,
                          perp="residual", joint=False, batch_size=2, log=None)
    after = _gate_mse(student, teacher, ids)
    assert after < before * 0.9, (before, after)
    assert rep["mean_rel_after"] < rep["mean_rel_before"]
    assert rep["mean_rel_deployed"] < rep["mean_rel_before"]
    mods = [m for m in student.modules() if isinstance(m, CoreLinear)]
    assert mods and any(float(m.z_perp.detach().norm()) > 0 for m in mods)
    assert all(torch.isfinite(m.z_col).all() for m in mods)


def test_calibrate_ridge_zero_perp_is_plain_init():
    H, d_ff, r, V = 32, 96, 16, 128
    teacher, student = _pair(H, d_ff, r, V)
    cfg = _cfg(r, ridge_perp="zero")
    _build_factorized_student(student, teacher, cfg, branch="core")
    ids = torch.randint(0, V, (4, 16))
    calibrate_ridge(student, teacher, ids, target_modules=cfg.target_modules,
                    perp="zero", joint=True, batch_size=2, log=None)
    mods = [m for m in student.modules() if isinstance(m, CoreLinear)]
    assert mods and all(float(m.z_perp.detach().norm()) == 0 for m in mods)
    for m in mods:
        if m.side == "left":
            assert torch.allclose(m.merged_weight(), m.basis_col @ m.z_col.T, atol=1e-5)
        else:
            assert torch.allclose(m.merged_weight(), m.z_col @ m.basis_col, atol=1e-5)


@pytest.mark.parametrize("warm_start", ["ridge_ckpt", "pt_ckpt", "base"])
def test_ridge_ckpt_builder_preserves_complement(warm_start):
    H, d_ff, r, nlayer = 24, 64, 12, 2
    torch.manual_seed(2)
    teacher = TinyLM(64, H, d_ff, nlayer)
    student = TinyLM(64, r, d_ff, nlayer)
    with torch.no_grad():
        for p in teacher.parameters():
            p.normal_(std=0.05)
    with torch.no_grad():
        for l in range(nlayer):
            tmlp = teacher.layers[l].mlp
            smlp = student.layers[l].mlp
            T, U = build_core_basis(tmlp.gate_proj.weight, "left")
            W = T @ (torch.randn(r, H) * 0.05).T + U @ (torch.randn(r, d_ff - H) * 0.02).T
            smlp.gate_proj.weight.copy_(W)
            T, U = build_core_basis(tmlp.up_proj.weight, "left")
            W = T @ (torch.randn(r, H) * 0.05).T + U @ (torch.randn(r, d_ff - H) * 0.02).T
            smlp.up_proj.weight.copy_(W)
            R, V = build_core_basis(tmlp.down_proj.weight, "right")
            W = (torch.randn(r, H) * 0.05) @ R + (torch.randn(r, d_ff - H) * 0.02) @ V
            smlp.down_proj.weight.copy_(W)
    expected = {f"layers.{l}.mlp.{n}": getattr(student.layers[l].mlp, n).weight.detach().clone()
                for l in range(nlayer) for n in ("gate_proj", "up_proj", "down_proj")}
    cfg = TwydLRCConfig(branch="core", student_hidden=r, warm_start=warm_start,
                        ridge_perp="residual",
                        target_modules=["gate_proj", "up_proj", "down_proj"])
    _build_factorized_student(student, teacher, cfg, branch="core")
    for name, W in expected.items():
        parent = student
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        mod = getattr(parent, parts[-1])
        assert torch.allclose(mod.merged_weight(), W, atol=1e-5), name


def test_calibrate_ridge_initializes_core_lora_factors():
    H, d_ff, r, V = 32, 96, 16, 128
    teacher, student = _pair(H, d_ff, r, V)
    cfg = _cfg(r, ridge_perp="residual", lora_rank=4, lora_alpha=16.0,
               lora_mode="core", freeze_col=True)
    _build_factorized_student(student, teacher, cfg, branch="core")
    ids = torch.randint(0, V, (4, 16))
    calibrate_ridge(student, teacher, ids, target_modules=cfg.target_modules,
                    perp="residual", joint=True, batch_size=2, log=None)
    mods = [m for m in student.modules() if isinstance(m, CoreLinearLoRA)]
    assert mods
    for m in mods:
        assert m.lora_A.shape[1] == 4 and m.lora_B.shape[1] == 4
        assert torch.isfinite(m.merged_weight()).all()
        assert not m.z_col.requires_grad  # freeze_col preserved
    assert any(float(m.lora_A.detach().norm()) > 0 for m in mods)
    # deploy path: CORE-LR factors fold into a standard .weight
    from twyd.merge import merged_state_dict
    sd = merged_state_dict(student)
    assert not any(("lora_" in k or ".z_col" in k or ".z_perp" in k or "basis_" in k)
                   for k in sd)
    assert any(k.endswith("gate_proj.weight") for k in sd)


def test_ridge_ckpt_dense_preserves_full_deployed_weight():
    H, d_ff, r = 24, 64, 12
    torch.manual_seed(3)
    teacher = TinyLM(64, H, d_ff, 1)
    student = TinyLM(64, r, d_ff, 1)
    W = torch.randn(d_ff, r)
    with torch.no_grad():
        student.layers[0].mlp.gate_proj.weight.copy_(W)
    cfg = TwydLRCConfig(branch="dense", student_hidden=r, warm_start="ridge_ckpt",
                        target_modules=["gate_proj"])
    _build_factorized_student(student, teacher, cfg, branch="dense")
    assert torch.allclose(student.layers[0].mlp.gate_proj.weight, W, atol=1e-6)


def test_load_calib_ids_pt_and_jsonl(tmp_path):
    p = tmp_path / "ids.pt"
    torch.save({"input_ids": torch.arange(300).view(3, 100)}, p)
    ids = load_calib_ids(str(p), seq_len=50, num_seqs=2)
    assert ids.shape == (2, 50)
    q = tmp_path / "ids.jsonl"
    q.write_text(json.dumps({"input_ids": list(range(40))}) + "\n")
    ids2 = load_calib_ids(str(q), seq_len=20, num_seqs=2)
    assert ids2.shape == (2, 20)
    assert pack_calib_ids(list(range(100)), seq_len=16, num_seqs=3).shape == (3, 16)


def test_prepare_model_impl_ridge_tiny_hf(tmp_path):
    """Integration gate: warm_start=ridge runs inside the ms-swift prepare path."""
    from transformers import AutoModelForCausalLM, LlamaConfig
    from twyd.tuner import CoreLRCAdapter, CoreLinear

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    teacher = AutoModelForCausalLM.from_config(cfg)
    teacher.save_pretrained(str(tmp_path))
    torch.save({"input_ids": torch.randint(0, 128, (4, 16))}, tmp_path / "calib.pt")

    class Args:
        pass

    args = Args()
    args.model = str(tmp_path)
    args.tuner_kwargs = {
        "teacher_model": str(tmp_path), "student_hidden": 16, "stage": "pt",
        "warm_start": "ridge", "ridge_perp": "residual",
        "master_p": "embed_svd",
        "calib_ids": str(tmp_path / "calib.pt"),
        "calib_num_seqs": 2, "calib_seq_len": 16, "calib_batch_size": 2,
        "calib_joint": False,
    }
    student = AutoModelForCausalLM.from_pretrained(str(tmp_path))
    wrapper = CoreLRCAdapter.prepare_model(args, student)
    assert getattr(wrapper, "twyd_ridge", None) is not None
    assert wrapper.twyd_ridge["n_groups"] == 2
    mods = [m for m in wrapper.student.modules() if isinstance(m, CoreLinear)]
    assert mods and any(float(m.z_perp.detach().norm()) > 0 for m in mods)
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out = wrapper.student(input_ids=ids)
    assert out.logits.shape == (2, 16, 128)


