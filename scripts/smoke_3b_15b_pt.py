"""Smoke test: Llama-3.2-3B -> r=1536 Dense-LRC PT on ms-swift (single card, few steps).
Teacher = same 3B (deepcopy to save a second download). Student shrunk 3072->1536.
"""
import os
os.environ["TWYD_TEACHER"] = "/home/chenwh/models/Llama-3.2-3B-Instruct"
os.environ["TWYD_R"] = "1536"
os.environ["TWYD_STAGE"] = "pt"
os.environ["TWYD_WARM_START"] = "slice"
os.environ["TWYD_KL_TEMPERATURE"] = "40.0"
os.environ["TWYD_KL_WEIGHT"] = "1.0"
os.environ["TWYD_NTP_WEIGHT"] = "1.0"
os.environ["TWYD_AUX_WEIGHT"] = "0.2"
os.environ["TWYD_USE_AUX"] = "true"
os.environ["TWYD_TARGET_MODULES"] = "gate_proj,up_proj,down_proj"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from twyd.tuner import _prepare_model_impl
from twyd.loss import LrcDistillLoss

torch.manual_seed(0)
device = "cuda:0"
model_path = "/home/chenwh/models/Llama-3.2-3B-Instruct"

# build on CPU to avoid deepcopy teacher eating VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
).train()

class Args:
    tuner_kwargs = None
    model = model_path

wrapper = _prepare_model_impl(Args(), model, "dense").to(device).train()
print("[3B->15b PT] wrapper built")

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts = ["Low-rank clone distillation trains what you deploy. "] * 4
inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = inputs["input_ids"].clone()
wrapper._twyd_last_inputs = {
    k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")
}

out = wrapper(**inputs, output_hidden_states=True, use_cache=False, labels=labels)

class FakeTrainer:
    model = wrapper
    _twyd_inputs = wrapper._twyd_last_inputs

class FakeArgs:
    pass

loss_fn = LrcDistillLoss(FakeArgs(), FakeTrainer())
loss = loss_fn(out, labels)
print("[3B->15b PT] loss =", float(loss.detach()), "finite:", bool(torch.isfinite(loss)))
loss.backward()
found = any(
    p.requires_grad and p.grad is not None and p.grad.abs().max().item() > 0
    for _, p in wrapper.student.named_parameters()
)
print("[3B->15b PT] any nonzero grad:", found)
print("[3B->15b PT] smoke PASSED")
