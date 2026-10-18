"""Standalone API smoke test for Core-LRC + merge export on ms-swift.
Verifies: model load -> Core-LRC tuner build -> forward -> loss -> backward -> merge_and_save.
Uses Llama-3.2-1B as both student & teacher (teacher resolved via deepcopy).
"""
import os
os.environ["TWYD_TEACHER"] = "/home/chenwh/models/Llama-3.2-1B-Instruct"
os.environ["TWYD_R"] = "1024"
os.environ["TWYD_STAGE"] = "pt"
os.environ["TWYD_WARM_START"] = "slice"
os.environ["TWYD_KL_TEMPERATURE"] = "40.0"
os.environ["TWYD_AUX_WEIGHT"] = "0.2"
os.environ["TWYD_USE_AUX"] = "true"
os.environ["TWYD_TARGET_MODULES"] = "gate_proj,up_proj,down_proj"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from twyd.tuner import _prepare_model_impl
from twyd.loss import LrcDistillLoss
from twyd.merge import merge_and_save, merged_state_dict

torch.manual_seed(0)
device = "cuda:0"
model_path = "/home/chenwh/models/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
).train()  # keep on CPU for build; move wrapper to GPU after

class Args:
    tuner_kwargs = None
    model = model_path

wrapper = _prepare_model_impl(Args(), model, "core").to(device).train()
print("[core] wrapper built")
n_train = 0
for n, p in wrapper.student.named_parameters():
    if p.requires_grad:
        n_train += 1
print("[core] total trainable tensors:", n_train)

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts = ["Hello world, this is a test.", "Core-LRC adds an orthogonal complement branch."] * 2
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
print("[core] loss =", float(loss.detach()), "finite:", bool(torch.isfinite(loss)))
loss.backward()

found = False
for n, p in wrapper.student.named_parameters():
    if p.requires_grad and p.grad is not None and p.grad.abs().max().item() > 0:
        print("[core] grad ok:", n, "max|grad|=", float(p.grad.abs().max()))
        found = True
        break
print("[core] any nonzero grad:", found)

# --- merge & save to a temp dir -------------------------------------------
out_dir = "/tmp/twyd_core_merged"
os.makedirs(out_dir, exist_ok=True)
merge_and_save(wrapper, out_dir, safe_serialization=True)
print("[core] merged saved to", out_dir)

# verify merged shapes match student original widths (d_ff x r / r x d_ff)
from safetensors.torch import load_file
sd = load_file(os.path.join(out_dir, "model.safetensors"))
for k in sd:
    if "mlp.gate_proj.weight" in k or "mlp.up_proj.weight" in k:
        assert sd[k].shape[1] == 1024, (k, sd[k].shape)  # d_ff x r
    if "mlp.down_proj.weight" in k:
        assert sd[k].shape[0] == 1024, (k, sd[k].shape)  # r x d_ff
print("[core] merged shapes verified (gate/up: d_ff x r; down: r x d_ff)")
print("[core] CORE-LRC + merge smoke test PASSED")
