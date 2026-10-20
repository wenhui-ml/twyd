"""merge_weight: T Z^T (+ U_perp Z_perp^T) -> standard HF weight (Prop.1).

Dense branch: nn.Linear weights pass through verbatim (already d_ff x r).
Core/plain: CoreLinear.merged_weight() rebuilds the deployed matrix.
"""
import json
import os

import torch
from safetensors.torch import save_file


def merged_state_dict(model) -> dict:
    student = getattr(model, "student", model)
    out = {}
    # Modules that expose merged_weight (CoreLinear, CoreLinearLoRA,
    # DenseLinearLoRA) fold all their trainable params (z_col/z_perp or
    # lora_A/lora_B) into one deployed weight; skip those params here so
    # they aren't saved twice or separately.
    merged_mod_names = set()
    for mod_name, mod in student.named_modules():
        if mod_name == "":
            continue
        if callable(getattr(mod, "merged_weight", None)):
            merged_mod_names.add(mod_name)
    for n, p in student.named_parameters():
        ln = n.lower()
        if ".z_col" in n or ".z_perp" in n or "basis_" in n:
            continue
        if "lora_a" in ln or "lora_b" in ln:
            mod_name = n.rsplit(".", 1)[0] if "." in n else ""
            if mod_name in merged_mod_names:
                continue
        mod_name = n.rsplit(".", 1)[0] if "." in n else ""
        if mod_name in merged_mod_names:
            continue
        out[n] = p.detach().cpu()
    for mod_name, mod in student.named_modules():
        mw = getattr(mod, "merged_weight", None)
        if callable(mw) and mod_name != "":
            with torch.no_grad():
                W = mw().detach().cpu()
            # find param name: mod_name + ".weight"
            key = mod_name + ".weight"
            out[key] = W
    # Dense branch has no merged_weight: its nn.Linear weights are already the
    # deployed d_ff x r matrix and were copied in the named_parameters loop above.
    # teacher / buffers excluded; keep student trainable-dense params
    for n, buf in student.named_buffers():
        if "basis_" in n:
            continue
    # Tied embeddings share storage; safetensors rejects shared tensors.
    seen = set()
    for k in list(out.keys()):
        v = out[k]
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        ptr = v.data_ptr()
        if ptr in seen:
            out[k] = v.detach().clone()
        else:
            seen.add(ptr)
    return out


def merge_and_save(model, save_directory: str, safe_serialization: bool = True) -> str:
    os.makedirs(save_directory, exist_ok=True)
    student = getattr(model, "student", model)
    sd = merged_state_dict(model)
    # hidden_size -> student width r
    cfg = getattr(student, "config", None)
    if cfg is not None:
        twcfg = getattr(model, "twyd_cfg", None)
        meta = getattr(model, "twyd_meta", {}) or {}
        rs = [v.get("r") for v in meta.values() if isinstance(v, dict) and v.get("r")]
        if rs:
            try:
                cfg.hidden_size = int(rs[0])
            except Exception:
                pass
        save_fn = getattr(student, "save_pretrained", None)
        # save config + tokenizer-agnostic files via HF, then overwrite weight file
        try:
            student.save_pretrained(save_directory, state_dict={} if False else None)
        except Exception:
            pass
        try:
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(json.loads(cfg.to_json_string()), f, indent=2)
        except Exception:
            pass
    save_file(sd, os.path.join(save_directory, "model.safetensors"))
    return save_directory
