"""Convert a twyd training checkpoint to a deployable HF student (merge-only).

Two merge paths:
  Path A: checkpoint already stores merged d_ff x r weights -> copy through.
  Path B: checkpoint stores z_col/z_perp factors (CoreLinear) -> rebuild via
          CoreLinear.merged_weight() using the teacher SVD basis, then save
          the standard deployed HF weights (Prop.1: shape identical to dense).

Usage:
  # Path A: swift already merged via Tuner.save_pretrained -> verbatim copy.
  python scripts/convert_ckpt.py --ckpt <swift_output_dir> --save <deploy_dir>

  # Path B: raw z_col/z_perp factors -> need --base (student base) + --teacher
  python scripts/convert_ckpt.py --ckpt <raw_factor_dir> --save <deploy_dir> \
      --base <student_hf> --teacher <teacher_hf> --branch core|dense|plain
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_sd(ckpt: str) -> dict:
    from safetensors.torch import load_file
    src = os.path.join(ckpt, "model.safetensors")
    if os.path.isfile(src):
        return load_file(src)
    src = os.path.join(ckpt, "pytorch_model.bin")
    if os.path.isfile(src):
        import torch
        return torch.load(src, map_location="cpu", weights_only=True)
    for name in os.listdir(ckpt):
        if name.endswith(".safetensors"):
            return load_file(os.path.join(ckpt, name))
    raise FileNotFoundError(f"no safetensors/bin found in {ckpt}")


def _path_b_merge(ckpt: str, save: str, base: str, teacher: str, branch: str):
    """Rebuild deployed weights from z_col/z_perp factors via CoreLinear."""
    import torch
    from transformers import AutoModelForCausalLM
    from twyd.tuner import _build_factorized_student, _TwydWrapper, _freeze_for_branch
    from twyd.config import TwydLRCConfig
    from twyd.merge import merge_and_save

    student = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", trust_remote_code=True)
    teacher_m = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype="bfloat16", trust_remote_code=True)
    teacher_m.eval()

    sd = _load_sd(ckpt)
    zcol_shapes = [v.shape for k, v in sd.items() if ".z_col" in k]
    if not zcol_shapes:
        raise ValueError("no .z_col tensors found in checkpoint; path B needs factor weights")
    r = int(zcol_shapes[0][0])

    cfg = TwydLRCConfig(
        branch=branch,
        teacher_model=teacher,
        student_hidden=r,
        stage="pt",
        warm_start="slice",
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    cfg.validate()

    meta = _build_factorized_student(student, teacher_m, cfg, branch=branch)
    wrapper = _TwydWrapper(student, teacher_m, cfg)
    wrapper.twyd_cfg = cfg
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)

    own = dict(student.named_parameters())
    loaded = 0
    with torch.no_grad():
        for k, v in sd.items():
            short = k.split("student.")[-1] if "student." in k else k
            key = short if short in own else (k if k in own else None)
            if key is None:
                continue
            p = own[key]
            if tuple(p.shape) == tuple(v.shape):
                p.copy_(v.to(p.dtype))
                loaded += 1
    print(f"[convert] folded {loaded} tensors from checkpoint into factorized student")

    os.makedirs(save, exist_ok=True)
    merge_and_save(wrapper, save, safe_serialization=True)
    print(f"[convert] merged deployed weights saved -> {save} (branch={branch}, r={r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training output dir (has adapter/merged weights)")
    ap.add_argument("--save", required=True, help="output deploy dir")
    ap.add_argument("--base", default=None, help="student base HF id/path (needed for path B)")
    ap.add_argument("--teacher", default=None, help="teacher HF id/path (needed for path B SVD basis)")
    ap.add_argument("--branch", default="core", choices=["core", "dense", "plain"],
                    help="twyD branch (path B only)")
    ap.add_argument("--stage", default="pt", choices=["pt", "sft"],
                    help="stage tag only (merge logic identical)")
    args = ap.parse_args()

    sd = _load_sd(args.ckpt)

    has_zcol = any(".z_col" in k for k in sd)
    has_zperp = any(".z_perp" in k for k in sd)

    if has_zcol or has_zperp:
        if not args.base or not args.teacher:
            raise ValueError(
                "checkpoint contains factor weights (.z_col/.z_perp); "
                "path B requires --base and --teacher"
            )
        _path_b_merge(args.ckpt, args.save, args.base, args.teacher, args.branch)
        return

    import torch
    from safetensors.torch import save_file
    os.makedirs(args.save, exist_ok=True)
    save_file(sd, os.path.join(args.save, "model.safetensors"))
    print(f"[convert] path A: copied {len(sd)} tensors -> {args.save}")
    if args.base:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
            with open(os.path.join(args.save, "config.json"), "w") as f:
                json.dump(json.loads(cfg.to_json_string()), f, indent=2)
        except Exception as e:
            print(f"[convert] config passthrough skipped: {e}")


if __name__ == "__main__":
    main()
