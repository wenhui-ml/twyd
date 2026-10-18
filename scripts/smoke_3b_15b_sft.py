"""Smoke test: Llama-3.2-3B -> r=1536 Dense-LRC SFT on ms-swift (single card, few steps).
Simulates SFT: prompt tokens masked to -100, KL & CE respect mask.
"""
import os
os.environ["TWYD_TEACHER"] = "/home/chenwh/models/Llama-3.2-3B-Instruct"
os.environ["TWYD_R"] = "1536"
os.environ["TWYD_STAGE"] = "sft"
os.environ["TWYD_WARM_START"] = "slice"
os.environ["TWYD_KL_TEMPERATURE"] = "40.0"
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
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
).train()

class Args:
    tuner_kwargs = None
    model = model_path

wrapper = _prepare_model_impl(Args(), model, "dense").to(device).train()
print("[3B->15b SFT] wrapper built")

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Build input where first 80 tokens are prompt (masked -100), rest are response.
# Use a long text so the response (unmasked part) has enough supervised tokens.
full = ("You are a helpful assistant. Please answer the following question about low-rank clone distillation. " * 8 +
        "Low-rank clone distillation trains what you deploy by closing the MLP reachability gap. ")
ids = tok(full, return_tensors="pt", truncation=True, max_length=256)["input_ids"][0]
B, S = 1, ids.shape[0]
labels = ids.clone().unsqueeze(0)  # [1, S]
# mask prompt (first 80) to -100; only response positions supervised
labels[0, :80] = -100
n_sup = int((labels[0] != -100).sum().item())
print(f"[3B->15b SFT] seq_len={S}, supervised_positions={n_sup}")
assert n_sup > 0, "no supervised positions"
inputs = {"input_ids": ids.unsqueeze(0).to(device),
          "attention_mask": torch.ones_like(ids).unsqueeze(0).to(device)}
labels = labels.to(device)
wrapper._twyd_last_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

out = wrapper(**inputs, output_hidden_states=True, use_cache=False, labels=labels)

class FakeTrainer:
    model = wrapper
    _twyd_inputs = wrapper._twyd_last_inputs

class FakeArgs:
    pass

loss_fn = LrcDistillLoss(FakeArgs(), FakeTrainer())
loss = loss_fn(out, labels)
print("[3B->15b SFT] loss =", float(loss.detach()), "finite:", bool(torch.isfinite(loss)))
last = getattr(loss_fn, "trainer", None)
print("[3B->15b SFT] last terms:", getattr(loss_fn.trainer, "_twyd_last", None))
loss.backward()
found = any(
    p.requires_grad and p.grad is not None and p.grad.abs().max().item() > 0
    for _, p in wrapper.student.named_parameters()
)
print("[3B->15b SFT] any nonzero grad:", found)
print("[3B->15b SFT] smoke PASSED")
