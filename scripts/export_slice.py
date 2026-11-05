"""Export the zero-training width-r student (the ``WARM_START=slice`` init).

Every twyd run builds this student inside ``_shrink_student_`` *before* a single
weight is learned: the teacher's residual stream is projected down to ``r`` dims
and the MLP/attention matrices are truncated accordingly.  Scoring the export
directly answers the capacity question that more tokens cannot:

    how much of the teacher's MMLU survives projecting 3072 -> r?

Sweeping r gives the MMLU-vs-rank curve (see docs/EXPERIMENTS.md §5).

    python scripts/export_slice.py --r 2048 --out output/slices/r2048
    PRESET=paper GPU=0 bash scripts/run_lm_eval.sh output/slices/r2048 paper_r2048
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from twyd.tuner import _attn_impl_of, _hidden_of, _shrink_student_  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/chenwh/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--r", type=int, required=True, help="target hidden size")
    ap.add_argument("--out", required=True, help="output dir (HF format)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--master-p", default="embed_svd", choices=["embed_svd", "none"],
                    help=("embed_svd = Paper Alg.1 (i): project the residual stream onto the "
                          "top-r directions of the tied embedding. 'none' keeps the raw "
                          "coordinate slice, which is NOT faithful (28 layers compound the "
                          "half-dot-product error): measured CE 12.46 vs teacher 2.86."))
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    teacher.eval()
    # The student starts as a second copy of the teacher; _shrink_student_ reads
    # the full-width weights out of `teacher` to build the projected model.
    student = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    h = _hidden_of(student)
    if args.r == h:
        shrunk = student
    elif args.r > h:
        raise SystemExit(f"--r {args.r} exceeds the teacher hidden size {h}")
    else:
        shrunk = _shrink_student_(student, teacher, args.r, attn_impl=_attn_impl_of(student))
        if args.master_p == "embed_svd":
            from twyd.init import build_master_P, project_student_with_P_
            P = build_master_P(teacher, args.r)
            n_proj = project_student_with_P_(shrunk, teacher, P)
            print(f"[slice] master P {tuple(P.shape)}: projected {n_proj} tensors", flush=True)

    os.makedirs(args.out, exist_ok=True)
    shrunk.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    n = sum(p.numel() for p in shrunk.parameters())
    print(f"[slice] h={h} -> r={args.r}  params={n / 1e6:.1f}M  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
