"""Unit tests for Dense-TWYD 2.0 (Manifold Alignment + Full-Bus Co-adaptation)."""
from types import SimpleNamespace
import torch
import torch.nn as nn

from twyd.config import TwydLRCConfig
from twyd.loss import LrcDistillLoss
from twyd.tuner import _freeze_for_branch, _TwydWrapper


class _MiniBlock(nn.Module):
    def __init__(self, d: int, d_ff: int, q_dim: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(d, q_dim, bias=False),
            "k_proj": nn.Linear(d, q_dim // 2, bias=False),
            "v_proj": nn.Linear(d, q_dim // 2, bias=False),
            "o_proj": nn.Linear(q_dim, d, bias=False),
        })
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(d, d_ff, bias=False),
            "up_proj": nn.Linear(d, d_ff, bias=False),
            "down_proj": nn.Linear(d_ff, d, bias=False),
        })

    def forward(self, x):
        h = self.input_layernorm(x)
        q = self.self_attn["q_proj"](h)
        a = self.self_attn["o_proj"](q)
        x = x + a
        m = self.post_attention_layernorm(x)
        g = torch.nn.functional.silu(self.mlp["gate_proj"](m)) * self.mlp["up_proj"](m)
        return x + self.mlp["down_proj"](g)


class _MiniCausalLM(nn.Module):
    def __init__(self, d: int, d_ff: int, q_dim: int, vocab: int = 128, n_layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_hidden_layers=n_layers, vocab_size=vocab)
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_MiniBlock(d, d_ff, q_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
        h = self.embed(input_ids)
        h_list = [h]
        for blk in self.layers:
            h = blk(h)
            h_list.append(h)
        logits = self.lm_head(self.norm(h))
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(h_list) if output_hidden_states else None,
        )


def test_topk_tail_bucket_kl_properties():
    loss_fn = LrcDistillLoss(SimpleNamespace(), SimpleNamespace())
    torch.manual_seed(0)
    B, S, V = 2, 8, 64
    s = torch.randn(B, S, V, requires_grad=True)
    t = torch.randn(B, S, V)

    # Zero when identical
    kl_zero = loss_fn._per_pos_kl(t, t, T=3.0, topk=8)
    assert torch.allclose(kl_zero, torch.zeros_like(kl_zero), atol=1e-5)

    # Strictly non-negative and differentiable when different
    kl_val = loss_fn._per_pos_kl(s, t, T=3.0, topk=8)
    assert (kl_val >= 0).all()
    kl_val.mean().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_rho_token_selection_weights():
    loss_fn = LrcDistillLoss(SimpleNamespace(), SimpleNamespace())
    torch.manual_seed(7)
    B, S, V = 1, 20, 32
    y = torch.randint(0, V, (B, S))
    # Teacher has low CE everywhere except last 2 tokens (noise)
    t_logits = torch.full((B, S, V), -5.0)
    t_logits.scatter_(-1, y.unsqueeze(-1), 5.0)
    t_logits[:, -2:, :] = torch.randn(B, 2, V) * 10.0  # noisy teacher positions

    # Student matches teacher except at positions 2..5 where student is wrong
    s_logits = t_logits.clone()
    s_logits[:, 2:6, :] = -t_logits[:, 2:6, :]
    per_pos_kl = loss_fn._per_pos_kl(s_logits, t_logits, T=3.0, topk=8)

    w = loss_fn._rho_token_weights(s_logits, t_logits, y, per_pos_kl, T=3.0,
                                  rho_keep=0.25, rho_min_weight=0.1)
    assert w.shape == (B, S)
    # Positions 2..5 (4 positions = 20% of 20) should all be selected with weight 1.0
    assert (w[:, 2:6] == 1.0).all()
    # Noisy teacher positions at the end should be filtered out (weight 0.1)
    assert (w[:, -2:] == torch.tensor(0.1, dtype=w.dtype)).all()


def test_full_bus_unfreeze_and_writer_decoupling():
    student = _MiniCausalLM(d=16, d_ff=32, q_dim=32, vocab=64, n_layers=4)
    teacher = _MiniCausalLM(d=32, d_ff=32, q_dim=32, vocab=64, n_layers=4)
    cfg = TwydLRCConfig(
        branch="dense",
        lora_layers="0-1",
        unfreeze_attn=True,
        unfreeze_norm=True,
        writer_lr_scale=0.25,
        writer_grad_scale=0.2,
        writer_max_norm=1.0,
    )
    wrapper = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(wrapper, cfg)

    # Layers 0-1 should have MLP + Attention + LayerNorm trainable; layers 2-3 frozen
    for n, p in student.named_parameters():
        if "layers.0." in n or "layers.1." in n:
            assert p.requires_grad, f"expected {n} trainable"
        elif "layers.2." in n or "layers.3." in n:
            assert not p.requires_grad, f"expected {n} frozen"

    # Writer hook clamps & scales down_proj / o_proj gradients
    ids = torch.randint(0, 64, (1, 6))
    out = wrapper(input_ids=ids)
    (out.logits.pow(2).sum() * 100.0).backward()
    down_g = student.layers[0].mlp["down_proj"].weight.grad
    assert down_g is not None
    # Clamped to max_norm=1.0 then scaled by 0.2 -> norm <= 0.2 + eps
    assert float(down_g.norm()) <= 0.21

    # Custom optimizer groups separate Writer (lr=2.5e-5) from Reader (lr=1e-4)
    groups = wrapper.create_optimizer_param_groups(lr=1e-4, weight_decay=0.01)
    assert groups is not None
    lrs = sorted({g["lr"] for g in groups})
    assert lrs == [2.5e-5, 1e-4]


def test_logit_lens_mid_layer_alignment_without_P():
    student = _MiniCausalLM(d=16, d_ff=32, q_dim=32, vocab=48, n_layers=4)
    teacher = _MiniCausalLM(d=32, d_ff=32, q_dim=32, vocab=48, n_layers=4)
    cfg = TwydLRCConfig(
        branch="dense",
        kl_temperature=3.0,
        kl_topk=16,
        rho_keep=0.5,
        use_aux=True,
        aux_weight=0.2,
        aux_loss_type="logit_lens",
        aux_layers="1,2",
    )
    wrapper = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(wrapper, cfg)

    trainer = SimpleNamespace(model=wrapper, _twyd_inputs={})
    loss_fn = LrcDistillLoss(SimpleNamespace(), trainer)

    ids = torch.randint(0, 48, (2, 12))
    out = wrapper(input_ids=ids, output_hidden_states=True)
    total = loss_fn(out, ids)
    assert torch.isfinite(total)
    assert trainer._twyd_last["aux"] is not None and trainer._twyd_last["aux"] >= 0.0
    total.backward()
    # Student layer 1 MLP receives gradient, while frozen lm_head does not
    assert student.layers[1].mlp["up_proj"].weight.grad is not None
    assert student.lm_head.weight.grad is None


def test_single_token_chunk_does_not_crash():
    """Regression guard: TRUNC=split can emit a 1-token tail chunk (shift_s.size(1)==0)."""
    student = _MiniCausalLM(d=16, d_ff=32, q_dim=32, vocab=48, n_layers=2)
    teacher = _MiniCausalLM(d=32, d_ff=32, q_dim=32, vocab=48, n_layers=2)
    cfg = TwydLRCConfig(branch="dense", kl_temperature=3.0, kl_topk=16, rho_keep=0.35, use_aux=False)
    wrapper = _TwydWrapper(student, teacher, cfg)
    trainer = SimpleNamespace(model=wrapper, _twyd_inputs={})
    loss_fn = LrcDistillLoss(SimpleNamespace(), trainer)

    # 1-token input -> shift_s has length 0
    ids1 = torch.randint(0, 48, (1, 1))
    out1 = wrapper(input_ids=ids1)
    l1 = loss_fn(out1, ids1)
    assert torch.isfinite(l1) and l1.requires_grad
    l1.backward()

    # 2-token input -> shift_s has length 1
    ids2 = torch.randint(0, 48, (1, 2))
    out2 = wrapper(input_ids=ids2)
    l2 = loss_fn(out2, ids2)
    assert torch.isfinite(l2) and l2.requires_grad
    l2.backward()

