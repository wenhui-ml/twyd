"""Greedy data-aware ridge calibration (paper Alg.1) for TWYD.

Fits the deployed width-r MLP weights layer by layer on a few hundred packed
sequences, using the student's own compressed activations and the teacher's
matched activations.  The fitted W is written back as

    Z_col = projection of W onto col(T) / row(W)
    Z_perp = residual of W in U_perp / V_perp   (optional)

so the result is a standard HF checkpoint plus a ``ridge_weights.pt`` that
``warm_start="ridge_ckpt"`` can load.  This is calibration-distribution
equivalence, not exact function equivalence: width r < H necessarily leaves a
compression residual, which Z_perp and later training address.

Usage:
  # build a width-r student from the teacher, calibrate, save merged HF dir
  python scripts/calib_ridge.py --teacher /path/to/teacher --r 1536 \
      --calib-ids /path/to/calib.pt --out outputs/ridge-r1536 \
      --branch core --perp residual

  # calibrate an existing width-r (or full-width) student
  python scripts/calib_ridge.py --teacher ... --student ... --r 1536 ...

  # math-only sanity check, no HF download
  python scripts/calib_ridge.py --synthetic
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def _synthetic_check():
    from twyd.core_basis import build_core_basis, check_basis
    from twyd.init import (build_outlier_aware_P, factorize_deployed,
                           pack_calib_ids, ridge_deployed, ridge_zcol)
    torch.manual_seed(0)
    H, d_ff, r, N = 32, 96, 16, 128
    T, Up = build_core_basis(torch.randn(d_ff, H), "left")
    assert check_basis(T, Up)["ok"]
    C = torch.randn(N, r)
    Wp = torch.randn(r, H)
    Z = ridge_zcol(C, C @ Wp, Wp, ridge_lambda=1e-3)
    assert torch.allclose(Z, Wp, atol=1e-3), "ridge should recover prior without residual"
    P, idx = build_outlier_aware_P(torch.randn(N, H) * 10, r, outlier_keep=4)
    assert P.shape == (r, H) and len(idx) == 4
    for i, ch in enumerate(idx):
        assert P[i, ch] == 1.0
    # deployed ridge: exact linear map
    Wtrue = torch.randn(d_ff, r)
    HT = C @ Wtrue.T
    W, st = ridge_deployed(C, HT, W_prior=torch.zeros_like(Wtrue), ridge_lambda=1e-6)
    assert torch.allclose(W, Wtrue, atol=1e-4), (W - Wtrue).abs().max()
    assert st["rel_err_after"] < 1e-2
    # factorization round-trip
    Wt = torch.randn(d_ff, H)
    T, Up = build_core_basis(Wt, "left")
    Zc = torch.randn(r, H)
    Zp = torch.randn(r, d_ff - H)
    Wfull = T @ Zc.T + Up @ Zp.T
    Zc2, Zp2, _ = factorize_deployed(Wfull, Wt, "left")
    assert torch.allclose(T @ Zc2.T + Up @ Zp2.T, Wfull, atol=1e-5)
    ids = pack_calib_ids(torch.arange(1000), seq_len=64, num_seqs=4)
    assert ids.shape == (4, 64)
    print("synthetic calib check OK: ridge + outlier-P + factorize + pack")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None, help="optional width-r/full-width HF student")
    ap.add_argument("--r", type=int, default=1536, help="student hidden width")
    ap.add_argument("--branch", default="core", choices=["dense", "core", "plain"])
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-mode", default="core", choices=["dense", "core"])
    ap.add_argument("--freeze-col", action="store_true")
    ap.add_argument("--calib-ids", default=None, help=".pt/.npy/.jsonl/.txt calibration ids")
    ap.add_argument("--calib-jsonl", default=None, help="alias for text/jsonl calibration")
    ap.add_argument("--num-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--perp", default="residual", choices=["zero", "residual", "lowrank"])
    ap.add_argument("--master-p", default="none", choices=["none", "embed_svd"],
                    help="optional embedding-SVD P projection before MLP ridge")
    ap.add_argument("--head-mode", default="select", choices=["select", "merge"],
                    help="P attention head mapping: select first heads or average groups")
    ap.add_argument("--calib-attn", action="store_true",
                    help="also ridge q/k/v/o with head-projected teacher targets")
    ap.add_argument("--joint", action="store_true", default=True)
    ap.add_argument("--greedy", dest="joint", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--teacher-device", default=None,
                    help="keep the teacher on cpu to save GPU memory (default: --device)")
    ap.add_argument("--out", default="/home/chenwh/Papers/twyd/outputs/ridge_init")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.synthetic or args.teacher is None:
        _synthetic_check()
        return
    if not (args.calib_ids or args.calib_jsonl):
        raise SystemExit("need --calib-ids or --calib-jsonl (see docstring)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from twyd.config import TwydLRCConfig
    from twyd.init import (build_master_P, calibrate_attention, calibrate_ridge,
                           collect_mlp_groups, load_calib_ids,
                           project_student_with_P_, _deployed_weight)
    from twyd.merge import merge_and_save
    from twyd.tuner import (_TwydWrapper, _build_factorized_student,
                            _freeze_for_branch, _hidden_of, _shrink_student_)

    dev = torch.device(args.device)
    tdev = torch.device(args.teacher_device) if args.teacher_device else dev
    dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
    tdtype = torch.float32 if tdev.type == "cpu" else torch.bfloat16
    print(f"[calib] student device={dev} dtype={dtype}; teacher device={tdev} dtype={tdtype}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=tdtype, trust_remote_code=True,
        low_cpu_mem_usage=True).to(tdev).eval()
    if args.student:
        student = AutoModelForCausalLM.from_pretrained(
            args.student, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True).to(dev)
    else:
        student = _shrink_student_(teacher, teacher, args.r)
    if _hidden_of(student) != args.r:
        student = _shrink_student_(student, teacher, args.r)
    student.to(device=dev, dtype=dtype)
    P = None
    if args.master_p != "none":
        P = build_master_P(teacher, args.r)
        n = project_student_with_P_(student, teacher, P, head_mode=args.head_mode)
        print(f"[calib] master P {tuple(P.shape)}: projected {n} tensors "
              f"(embed/attn/mlp/norm)")
    cfg = TwydLRCConfig(
        branch=args.branch, teacher_model=args.teacher, student_hidden=args.r,
        stage="pt", warm_start="ridge", ridge_perp=args.perp,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_mode=args.lora_mode, freeze_col=bool(args.freeze_col))
    cfg.validate()
    meta = _build_factorized_student(student, teacher, cfg, branch=args.branch)
    wrapper = _TwydWrapper(student, teacher, cfg)
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    except Exception as e:
        print(f"[calib] tokenizer unavailable ({e}); only pre-tokenized ids are supported")
    path = args.calib_ids or args.calib_jsonl
    ids = load_calib_ids(path, tokenizer=tok, seq_len=args.seq_len,
                         num_seqs=args.num_seqs)
    print(f"[calib] ids={tuple(ids.shape)} from {path}")
    rep_attn = None
    if args.calib_attn:
        P_attn = P if P is not None else torch.eye(
            teacher.get_input_embeddings().weight.shape[1])[:args.r]
        rep_attn = calibrate_attention(
            wrapper.student, teacher, ids, P=P_attn, head_mode=args.head_mode,
            ridge_lambda=cfg.ridge_lambda, batch_size=args.batch_size)
        print(f"[calib] attn groups={rep_attn['n_groups']} "
              f"rel {rep_attn['mean_rel_before']:.4f} -> "
              f"{rep_attn['mean_rel_deployed']:.4f}")
    rep = calibrate_ridge(wrapper.student, teacher, ids,
                          target_modules=cfg.target_modules,
                          ridge_lambda=cfg.ridge_lambda, perp=args.perp,
                          joint=args.joint, batch_size=args.batch_size)
    print(f"[calib] groups={rep['n_groups']} tokens={rep['n_tokens']} "
          f"rel {rep['mean_rel_before']:.4f} -> "
          f"{rep.get('mean_rel_deployed', rep['mean_rel_after']):.4f}")
    if args.dry_run:
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))
        return

    os.makedirs(args.out, exist_ok=True)
    # Remove stale shards/index from a previous save so the merged
    # model.safetensors we write below is the one HF loads.
    for fn in os.listdir(args.out):
        if fn == "model.safetensors.index.json" or (fn.startswith("model-") and fn.endswith(".safetensors")):
            os.remove(os.path.join(args.out, fn))
    weights = {}
    if P is not None:
        # P changes embedding/attention/norm too, so save the full width-r
        # deployed checkpoint (not just MLP) for ridge_ckpt/diagnostics.
        for name, mod in wrapper.student.named_modules():
            if not name or not hasattr(mod, "weight"):
                continue
            w = mod.weight
            if not isinstance(w, torch.Tensor):
                continue
            if callable(getattr(mod, "merged_weight", None)):
                weights[name] = _deployed_weight(mod).detach().cpu()
            else:
                weights[name] = w.detach().cpu()
    else:
        for g in collect_mlp_groups(wrapper.student, teacher, cfg.target_modules):
            for short, (s_mod, _t_mod) in g["projs"].items():
                weights[f"{g['base']}.{short}"] = _deployed_weight(s_mod).detach().cpu()
    payload = {"weights": weights, "meta": rep}
    if rep_attn is not None:
        payload["attn"] = rep_attn
    if P is not None:
        payload["P"] = P.detach().cpu()
    torch.save(payload, os.path.join(args.out, "ridge_weights.pt"))
    with open(os.path.join(args.out, "ridge_report.json"), "w") as f:
        json.dump({"mlp": rep, "attn": rep_attn}, f, indent=2)
    if tok is not None:
        tok.save_pretrained(args.out)
    merge_and_save(wrapper, args.out)
    stale = os.path.join(args.out, "model.safetensors.index.json")
    if os.path.isfile(stale):
        os.remove(stale)
    print(f"[calib] wrote merged HF model + ridge_weights.pt to {args.out}")


if __name__ == "__main__":
    main()
