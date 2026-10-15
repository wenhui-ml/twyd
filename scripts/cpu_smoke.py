"""CPU smoke: tiny real HF forward, zero download (random init from config).

Builds tiny Llama teacher (H) + student (r) from LlamaConfig on CPU,
wraps all 3 branches via _build_factorized_student/_TwydWrapper/_freeze_for_branch,
runs real model forward + LrcDistillLoss + 10 Adam steps per branch.

Checks: forward ok, KL/NTP finite, core Z_perp leaves 0, dense weight moves,
plain trainable = r*H only, merged_state_dict deployable shapes.
"""
import sys

sys.path.insert(0, "/home/chenwh/Papers/twyd")

import torch

from twyd.config import TwydLRCConfig
from twyd.loss import LrcDistillLoss
from twyd.merge import merged_state_dict
from twyd.tuner import (
    _build_factorized_student,
    _freeze_for_branch,
    _TwydWrapper,
    CoreLinear,
)


def _tiny_pair(H=64, d_ff=128, r=32, layers=2, heads=4, vocab=256):
    from transformers import LlamaConfig, LlamaForCausalLM

    tcfg = LlamaConfig(hidden_size=H, intermediate_size=d_ff,
                       num_hidden_layers=layers, num_attention_heads=heads,
                       num_key_value_heads=2, vocab_size=vocab)
    scfg = LlamaConfig(hidden_size=r, intermediate_size=d_ff,
                       num_hidden_layers=layers, num_attention_heads=max(2, heads // 2),
                       num_key_value_heads=2, vocab_size=vocab)
    torch.manual_seed(0)
    teacher = LlamaForCausalLM(tcfg).float().eval()
    student = LlamaForCausalLM(scfg).float()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, student


def _run_branch(branch, steps=10):
    H, d_ff, r, vocab = 64, 128, 32, 256
    teacher, student = _tiny_pair(H, d_ff, r)
    cfg = TwydLRCConfig(branch=branch, student_hidden=r,
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        kl_temperature=2.0, kl_weight=1.0, ntp_weight=1.0,
                        aux_weight=0.0, use_aux=False)
    _build_factorized_student(student, teacher, cfg, branch=branch)
    w = _TwydWrapper(student, teacher, cfg)
    _freeze_for_branch(w, cfg)

    n_train = sum(p.numel() for p in w.student.parameters() if p.requires_grad)
    print(f"[{branch}] trainable params: {n_train}", flush=True)

    zps = [m.z_perp for m in w.student.modules() if isinstance(m, CoreLinear) and m.z_perp is not None]
    if zps:
        assert all(bool((z == 0).all()) for z in zps), "Z_perp must start at 0"

    opt = torch.optim.AdamW([p for p in w.student.parameters() if p.requires_grad], lr=1e-3)
    B, S = 2, 32
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (B, S))

    class A:
        tuner_kwargs = {"kl_temperature": 2.0, "kl_weight": 1.0,
                        "ntp_weight": 1.0, "aux_weight": 0.0, "use_aux": False}

    class FakeTrainer:
        def __init__(self, model):
            self.model = model
            self._twyd_inputs = {}

    tr = FakeTrainer(w)
    loss_fn = LrcDistillLoss(A(), tr)
    first, last = None, None
    for it in range(steps):
        opt.zero_grad()
        out = w.student(input_ids=ids)
        w._twyd_last_inputs = {"input_ids": ids}
        labels = ids.clone()
        val = loss_fn(out, labels)
        assert torch.isfinite(val), (branch, val)
        kl, ntp = tr._twyd_last["kl"], tr._twyd_last["ntp"]
        assert kl is not None and ntp is not None, tr._twyd_last
        if first is None:
            first = (float(kl), float(ntp), float(val.detach()))
        last = (float(kl), float(ntp), float(val.detach()))
        val.backward()
        opt.step()
    print(f"[{branch}] first KL/NTP/loss={first} last={last}", flush=True)
    if zps:
        moved = any(bool((z != 0).any()) for z in zps)
        print(f"[{branch}] Z_perp moved: {moved}", flush=True)
        assert moved, "core Z_perp must leave 0 after real steps"
    sd = merged_state_dict(w)
    for n, t in sd.items():
        if "gate_proj" in n or "up_proj" in n:
            assert tuple(t.shape) == (d_ff, r), (branch, n, tuple(t.shape))
        if "down_proj" in n:
            assert tuple(t.shape) == (r, d_ff), (branch, n, tuple(t.shape))
    print(f"[{branch}] CPU smoke OK ({len(sd)} merged tensors)", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default="dense,core,plain")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    for b in [x.strip() for x in args.branches.split(",") if x.strip()]:
        _run_branch(b, steps=args.steps)
    print("ALL CPU SMOKE OK")


if __name__ == "__main__":
    main()
