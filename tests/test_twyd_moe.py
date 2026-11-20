"""Unit tests for Phase 2 TWYD-MoE (r=1536 Shared-Dense + Routed Knowledge Experts)."""
from copy import deepcopy
from types import SimpleNamespace
import os
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from twyd.hf_moe.modeling_twyd_moe import TwydSharedRoutedMoE, inject_twyd_moe_
from twyd.loss import LrcDistillLoss
from twyd.moe import MoeLRCAdapter, save_twyd_moe_pretrained


def _tiny_llama(hidden: int = 32, d_ff: int = 64, layers: int = 4, vocab: int = 64):
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=d_ff,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def test_step0_bit_exact_identity():
    mother = _tiny_llama().eval()
    student = deepcopy(mother).eval()
    inject_twyd_moe_(student, moe_layers=[1, 2], num_experts=4, top_k=2, moe_intermediate_size=24)

    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        z_mother = mother(input_ids=ids).logits
        z_moe = student(input_ids=ids).logits
    assert torch.equal(z_mother, z_moe), f"max diff={(z_mother - z_moe).abs().max().item()}"


def test_moe_grad_checkpointing_and_router_aux_grad(monkeypatch):
    mother = _tiny_llama()
    monkeypatch.setenv("TWYD_TEACHER", "")
    monkeypatch.setenv("TWYD_R", "32")
    monkeypatch.setenv("TWYD_MOE_LAYERS", "1-2")
    monkeypatch.setenv("TWYD_MOE_EXPERTS", "4")
    monkeypatch.setenv("TWYD_MOE_TOPK", "2")
    monkeypatch.setenv("TWYD_MOE_INTER", "24")
    monkeypatch.setenv("TWYD_MOE_ROUTER_LR_SCALE", "2.0")
    monkeypatch.setenv("TWYD_WRITER_LR_SCALE", "0.5")
    monkeypatch.setenv("TWYD_T", "1.0")
    monkeypatch.setenv("TWYD_KL_TOPK", "16")
    monkeypatch.setenv("TWYD_RHO_KEEP", "0.5")

    wrapper = MoeLRCAdapter.prepare_model(
        SimpleNamespace(model="self_fake", gradient_accumulation_steps=8), mother
    )
    wrapper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    wrapper.train()

    moe1 = wrapper.student.model.layers[1].mlp
    assert isinstance(moe1, TwydSharedRoutedMoE)
    assert abs(moe1.aux_grad_scale - 0.125) < 1e-9
    # Orthogonal w_gate / w_up row L2 norms should be ~0.8 (matching Llama init scale, not 1/sqrt(d)).
    row_norm = moe1.w_gate[0].norm(dim=-1).mean().item()
    assert 0.6 < row_norm < 1.1, f"unexpected w_gate row norm: {row_norm}"

    trainer = SimpleNamespace(model=wrapper, _twyd_inputs={})
    loss_fn = LrcDistillLoss(SimpleNamespace(), trainer)

    # First backward: w_down is zero so w_down gets grad from CE/KL and router gets grad from _AttachAuxLoss!
    ids = torch.randint(0, 64, (2, 12))
    out = wrapper(input_ids=ids)
    loss = loss_fn(out, ids)
    assert torch.isfinite(loss)
    loss.backward()

    assert moe1.router.weight.grad is not None and moe1.router.weight.grad.abs().sum() > 0
    assert moe1.w_down.grad is not None and moe1.w_down.grad.abs().sum() > 0
    # Shared mother remains strictly frozen
    assert moe1.shared_mlp.gate_proj.weight.grad is None
    assert wrapper.student.model.layers[0].mlp.gate_proj.weight.grad is None

    # Optimizer param groups separate router (2e-4), w_down (5e-5), and w_gate/w_up (1e-4)
    groups = wrapper.create_optimizer_param_groups(lr=1e-4, weight_decay=0.01)
    lrs = sorted({g["lr"] for g in groups})
    assert lrs == [5e-5, 1e-4, 2e-4], f"unexpected lrs: {lrs}"


def test_moe_save_and_hf_trust_remote_code_reload(tmp_path, monkeypatch):
    mother = _tiny_llama()
    monkeypatch.setenv("TWYD_R", "32")
    monkeypatch.setenv("TWYD_MOE_LAYERS", "1-2")
    monkeypatch.setenv("TWYD_MOE_EXPERTS", "4")
    monkeypatch.setenv("TWYD_MOE_TOPK", "2")
    monkeypatch.setenv("TWYD_MOE_INTER", "24")

    wrapper = MoeLRCAdapter.prepare_model(SimpleNamespace(model="self_fake"), mother)
    # Give w_down non-zero weights to ensure routed path actually contributes and round-trips!
    with torch.no_grad():
        for mod in wrapper.student.modules():
            if isinstance(mod, TwydSharedRoutedMoE):
                mod.w_down.normal_(mean=0.0, std=0.05)

    wrapper.student.eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        z_before = wrapper.student(input_ids=ids).logits

    out_dir = str(tmp_path / "twyd_moe_ckpt")
    MoeLRCAdapter.save_pretrained(wrapper, out_dir)
    assert os.path.isfile(os.path.join(out_dir, "model.safetensors"))
    assert os.path.isfile(os.path.join(out_dir, "configuration_twyd_moe.py"))
    assert os.path.isfile(os.path.join(out_dir, "modeling_twyd_moe.py"))

    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
    with torch.no_grad():
        z_after = reloaded(input_ids=ids).logits
    assert torch.allclose(z_before, z_after, atol=1e-5), f"max diff={(z_before - z_after).abs().max().item()}"

    # Also verify SFT / continuation warm-start via MoeLRCAdapter.prepare_model(pt_ckpt=out_dir)
    # preserves all shared + routed weights (including non-zero w_down) bit-exactly!
    fresh_dense = _tiny_llama()
    monkeypatch.setenv("TWYD_STAGE", "sft")
    monkeypatch.setenv("TWYD_WARM_START", "pt_ckpt")
    monkeypatch.setenv("TWYD_PT_CKPT", out_dir)
    resumed = MoeLRCAdapter.prepare_model(SimpleNamespace(model="self_fake"), fresh_dense)
    resumed.student.eval()
    with torch.no_grad():
        z_resumed = resumed.student(input_ids=ids).logits
    assert torch.allclose(z_before, z_resumed, atol=1e-5), f"max diff={(z_before - z_resumed).abs().max().item()}"
    assert resumed.student.model.layers[1].mlp.w_down.abs().sum() > 0
