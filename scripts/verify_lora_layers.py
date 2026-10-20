"""Verify lora_layers + lora_mode selection and merge shapes (tiny CPU model)."""
import types
import torch
from torch import nn
from twyd.config import TwydLRCConfig
from twyd import tuner as T
from twyd.tuner import _TwydWrapper, _freeze_for_branch
from twyd.merge import merged_state_dict


class MLP(nn.Module):
    def __init__(self, h, ff):
        super().__init__()
        self.gate_proj = nn.Linear(h, ff, bias=False)
        self.up_proj = nn.Linear(h, ff, bias=False)
        self.down_proj = nn.Linear(ff, h, bias=False)


class Stu(nn.Module):
    def __init__(self, h=64, ff=128, nl=4):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=h, num_hidden_layers=nl,
                                            num_attention_heads=8, num_key_value_heads=4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(nl):
            b = nn.Module()
            b.mlp = MLP(h, ff)
            self.model.layers.append(b)


def run(branch, lora_mode, layers):
    stu, tea = Stu(), Stu()
    cfg = TwydLRCConfig(branch=branch, student_hidden=32,
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        lora_rank=8, lora_alpha=16.0,
                        lora_mode=lora_mode, lora_layers=layers)
    meta = T._build_factorized_student(stu, tea, cfg, branch=branch)
    w = _TwydWrapper(stu, tea, cfg)
    _freeze_for_branch(w, cfg)
    tr = sorted(n for n, p in w.named_parameters() if p.requires_grad)
    kinds = {}
    for n in tr:
        k = "lora" if "lora_" in n.lower() else ("z" if ("z_col" in n or "z_perp" in n) else "denseW")
        kinds[k] = kinds.get(k, 0) + 1
    sd = merged_state_dict(w)
    print(f"[{branch}/{lora_mode}/layers={layers!r}] trainable={len(tr)} kinds={kinds} merged={len(sd)}")
    for n in tr[:6]:
        print("   ", n)
    exp_layers = {0, 1} if layers else set(range(4))
    for n in tr:
        if "lora_" in n.lower():
            assert ".layers.0." in n or ".layers.1." in n, f"LoRA leaked outside 0-1: {n}"
    assert len(sd) == 12, sd.keys()
    print("    OK: LoRA confined to layers 0-1, merge shapes sane")


run("dense", "dense", "0-1")
run("core", "core", "0-1")
run("core", "dense", "0-1")
print("ALL OK")
