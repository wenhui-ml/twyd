"""merge_weight: T Z^T (+ U_perp Z_perp^T) -> standard HF weight (Prop.1)."""
import json
import os

import torch
from safetensors.torch import save_file


def merged_state_dict(model) -> dict:
    student = getattr(model, "student", model)
    out = {}
    for n, p in student.named_parameters():
        if ".z_col" in n or ".z_perp" in n or "basis_" in n:
            continue
        out[n] = p.detach().cpu()
    for mod_name, mod in student.named_modules():
        mw = getattr(mod, "merged_weight", None)
        if callable(mw) and not mod_name == "":
            with torch.no_grad():
                W = mw().detach().cpu()
            # find param name: mod_name + ".weight"
            key = mod_name + ".weight"
            out[key] = W
    # teacher / buffers excluded; keep student trainable-dense params
    for n, buf in student.named_buffers():
        if "basis_" in n:
            continue
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
