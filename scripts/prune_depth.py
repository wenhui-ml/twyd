"""Depth-prune the teacher at full width (keep hidden=3072, drop whole layers).

Why: width halving (`_shrink_student_`) is a *rank* projection - every layer's MLP
becomes rank <= r, and the measured cost is 12 pt of MMLU (docs/EXPERIMENTS.md 4c/5).
Depth pruning keeps every surviving layer full rank and only removes computation,
which the pruning literature (Minitron, ShortGPT, LaCo) reports as far cheaper for
knowledge retention at the same parameter count. This script builds the candidate
without any training so the *floor* can be scored first:

    python scripts/prune_depth.py --keep 0,1,2,3,24,25,26,27 --out output/pruned/p14
    PRESET=paper GPU=0 bash scripts/run_lm_eval.sh output/pruned/p14 paper_p14
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_keep(spec: str, n: int) -> list[int]:
    """`--keep` is either a count (evenly spaced) or an explicit list of indices."""
    if "," in spec:
        idx = sorted({int(x) for x in spec.split(",") if x.strip() != ""})
    else:
        k = int(spec)
        if k <= 0 or k > n:
            raise SystemExit(f"--keep {k} out of range for {n} layers")
        # keep the first and last layers, spread the rest evenly: the first block
        # builds the representation and the last block maps it to the LM head.
        idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    if not idx:
        raise SystemExit("empty layer list")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/chenwh/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--keep", required=True,
                    help="layer count (evenly spaced) or explicit comma list, e.g. 0,1,2,3,24,25,26,27")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    model.eval()
    layers = model.model.layers
    n = len(layers)
    keep = parse_keep(args.keep, n)
    model.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    # Each surviving attention block keeps the layer_idx it was built with; leaving
    # them stale breaks KV-cache indexing (IndexError) and any per-layer behaviour,
    # so renumber them to the new positions.
    for new_i, mod in enumerate(model.model.layers):
        if hasattr(mod, "self_attn"):
            mod.self_attn.layer_idx = new_i
    # Keep the new indices consistent for anything reading per-layer config lists.
    for attr in ("layer_types", "mlp_only_layers", "no_rope_layers"):
        seq = getattr(model.config, attr, None)
        if isinstance(seq, (list, tuple)) and len(seq) == n:
            setattr(model.config, attr, [seq[i] for i in keep])
    # position ids / rope are per-token, not per-layer: nothing else to fix.

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[prune] kept {len(keep)}/{n} layers {keep}  params={n_par / 1e6:.1f}M  -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
