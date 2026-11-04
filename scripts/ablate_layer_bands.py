#!/usr/bin/env python
"""Layer-band ablation: apply an arm's ΔW only to a subset of its layers.

Why: the 9-layer plan is the only effect in our grid that clears the measurement
floor (28 layers 0.6433 -> 9 layers 0.6594).  If one of the three bands carries the
whole effect, the product recipe collapses from 9 layers to 3 and the *forgetting*
story gets a depth-localised explanation.  No training: we recombine existing weights.

For every band we write a full standard checkpoint (base weights + the arm's ΔW on
that band's gate/up), so the same eval harness applies unchanged.

  python scripts/ablate_layer_bands.py \
      --base /home/chenwh/models/LRC-1.5B-Base \
      --arm  output/product/corelr-k64/.../checkpoint-2000 \
      --out  output/ablation \
      --bands 0-2 11-13 25-27 all
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COPY = ("config.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def parse_bands(specs):
    out = []
    for s in specs:
        if s == "all":
            out.append(("all", None))
            continue
        layers = set()
        for part in s.split(","):
            if "-" in part:
                a, b = part.split("-")
                layers |= set(range(int(a), int(b) + 1))
            else:
                layers.add(int(part))
        out.append((s, layers))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", nargs="+", default=["0-2", "11-13", "25-27", "all"])
    ap.add_argument("--modules", default="gate_proj,up_proj")
    args = ap.parse_args()

    mods = [m.strip() for m in args.modules.split(",") if m.strip()]
    base_f = os.path.join(args.base, "model.safetensors")
    arm_f = os.path.join(args.arm, "model.safetensors")
    for f in (base_f, arm_f):
        if not os.path.isfile(f):
            sys.exit(f"!! {f} missing")
    B = safe_open(base_f, "pt")
    A = safe_open(arm_f, "pt")
    keys = list(B.keys())

    # Which tensors did the arm actually change, and by how much (sanity).
    touched = {}
    for k in keys:
        if ".mlp." not in k or not k.endswith(".weight"):
            continue
        d = A.get_tensor(k).float() - B.get_tensor(k).float()
        rel = d.norm().item() / max(B.get_tensor(k).float().norm().item(), 1e-9)
        if rel > 0.01:
            touched[k] = rel
    print(f"[ablate] arm changed {len(touched)} MLP tensors (rel>1%)")

    os.makedirs(args.out, exist_ok=True)
    for name, layers in parse_bands(args.bands):
        dst = os.path.join(args.out, f"band-{name}")
        os.makedirs(dst, exist_ok=True)
        for f in COPY:
            src = os.path.join(args.base, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, f))
        out, n_swapped = {}, 0
        for k in keys:
            take_arm = False
            if layers is None:
                take_arm = k in touched
            else:
                for L in layers:
                    for m in mods:
                        if k == f"model.layers.{L}.mlp.{m}.weight":
                            take_arm = True
            if take_arm and k in A.keys():
                out[k] = A.get_tensor(k)
                n_swapped += 1
            else:
                out[k] = B.get_tensor(k)
        save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
        gb = os.path.getsize(os.path.join(dst, "model.safetensors")) / 2**30
        print(f"[ablate] band {name:8s}: swapped {n_swapped:3d} tensors -> {dst} ({gb:.2f} GiB)")
        del out
    print("[ablate] done")


if __name__ == "__main__":
    main()
