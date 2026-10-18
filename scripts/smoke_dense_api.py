"""Standalone API smoke test for Dense-LRC on ms-swift (no swift CLI).
Verifies: model load -> tuner build -> forward -> LrcDistillLoss -> backward.
Uses Llama-3.2-1B as both student & teacher (teacher resolved via deepcopy).
"""
import os
os.environ["TWYD_TEACHER"] = "/home/chenwh/models/Llama-3.2-1B-Instruct"
os.environ["TWYD_R"] = "1024"
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

model_path = "/home/chenwh/models/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).train()

class Args:
    tuner_kwargs = None
    model = model_path

args = Args()
wrapper = _prepare_model_impl(args, model, "dense")
wrapper = wrapper.to(device).train()
print("[smoke] wrapper built, trainable params:")
n_train = 0
for n, p in wrapper.student.named_parameters():
    if p.requires_grad:
        n_train += 1
        print("  ", n, tuple(p.shape))
print("[smoke] total trainable tensors:", n_train)

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts = ["Hello world, this is a test.", "Low-rank clone distillation trains what you deploy."] * 2
inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = inputs["input_ids"].clone()

# cache raw inputs for the loss (same role as TwydInputCacheCallback)
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
print("[smoke] loss =", float(loss.detach()), "finite:", bool(torch.isfinite(loss)))
loss.backward()

# report first non-empty grad
found = False
for n, p in wrapper.student.named_parameters():
    if p.requires_grad and p.grad is not None and p.grad.abs().max().item() > 0:
        print("[smoke] grad ok:", n, "max|grad|=", float(p.grad.abs().max()))
        found = True
        break
print("[smoke] any nonzero grad:", found)
print("[smoke] DENSE-LRC API smoke test PASSED")
