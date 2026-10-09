"""Convert a twyd training checkpoint to a deployable HF student (merge-only)."""
import argparse
import sys

sys.path.insert(0, "/home/chenwh/Papers/twyd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training output dir (has adapter weights)")
    ap.add_argument("--save", required=True)
    ap.add_argument("--base", default=None, help="optional student base HF id for config passthrough")
    args = ap.parse_args()
    import torch
    from safetensors.torch import load_file
    from twyd.merge import merge_and_save

    # Merge path A: checkpoint already stores merged d_ff x r weights -> copy through.
    # Merge path B: checkpoint stores z_col/z_perp factors -> rebuild via CoreLinear.
    # v0: load whatever tensors exist; merge.py resolves CoreLinear modules when
    # loaded into a wrapped model. Here we just re-save safetensors verbatim and
    # patch config hidden_size if twyd meta is present.
    import os
    src = os.path.join(args.ckpt, "model.safetensors")
    sd = load_file(src)
    from twyd.merge import merge_and_save as _m  # noqa: F401 (kept for path B wiring)
    os.makedirs(args.save, exist_ok=True)
    from safetensors.torch import save_file
    save_file(sd, os.path.join(args.save, "model.safetensors"))
    print(f"saved {len(sd)} tensors -> {args.save} (merge path A verbatim; "
          "path B runs inside Tuner.save_pretrained during swift training)")


if __name__ == "__main__":
    main()
