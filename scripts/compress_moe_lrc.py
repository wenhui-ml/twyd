#!/usr/bin/env python3
"""Compress a native HuggingFace MoE model (e.g. OLMoE-1B-7B) via LRC-MoE.

Examples:
  # Naive frequency pruning baseline (64 -> 32 experts)
  python scripts/compress_moe_lrc.py \
    --model ~/models/OLMoE-1B-7B-0125-Instruct \
    --out output/moe_lrc/olmoe-freq-e32 \
    --target-experts 32 --method freq --router-mode slice --no-lrc-ridge

  # Full LRC-MoE compression (64 -> 32 experts, Gram-OBS + KL-Router + Coupled LRC Ridge)
  python scripts/compress_moe_lrc.py \
    --model ~/models/OLMoE-1B-7B-0125-Instruct \
    --out output/moe_lrc/olmoe-lrc-e32 \
    --target-experts 32 --method lrc_cluster --router-mode kl_fit --lrc-ridge
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from twyd.moe_compress import MoeCompressConfig, compress_moe_model


def load_calib_batches(
    tokenizer,
    calib_files: List[str],
    num_seqs: int,
    seq_len: int,
    batch_size: int = 2,
) -> List[torch.Tensor]:
    """Build fixed-length calibration token batches from JSONL files (`text` field)."""
    texts: List[str] = []
    per_file = max(1, (num_seqs * 4) // max(len(calib_files), 1))
    for path in calib_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        got = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if got >= per_file:
                    break
                obj = json.loads(line)
                t = obj.get("text") or obj.get("content") or ""
                if len(t) >= 200:
                    texts.append(t)
                    got += 1

    # Interleave texts across files if multiple files given
    token_buf: List[int] = []
    seqs: List[torch.Tensor] = []
    eos = tokenizer.eos_token_id
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        token_buf.extend(ids)
        if eos is not None:
            token_buf.append(int(eos))
        while len(token_buf) >= seq_len and len(seqs) < num_seqs:
            seqs.append(torch.tensor(token_buf[:seq_len], dtype=torch.long))
            token_buf = token_buf[seq_len:]
        if len(seqs) >= num_seqs:
            break

    if len(seqs) < num_seqs:
        raise RuntimeError(
            f"Only assembled {len(seqs)}/{num_seqs} sequences of length {seq_len} from {calib_files}"
        )

    batches = []
    for i in range(0, len(seqs), batch_size):
        batches.append(torch.stack(seqs[i : i + batch_size], dim=0))
    return batches


@torch.no_grad()
def eval_heldout_ce(
    model,
    tokenizer,
    holdout_file: str,
    num_seqs: int = 32,
    seq_len: int = 512,
    device: torch.device | str = "cuda",
) -> dict:
    """Compute held-out negative log-likelihood (CE) and perplexity."""
    batches = load_calib_batches(tokenizer, [holdout_file], num_seqs=num_seqs, seq_len=seq_len, batch_size=2)
    total_nll = 0.0
    total_toks = 0
    for b in batches:
        ids = b.to(device=device)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[:, :-1, :].to(torch.float32)
        labels = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_toks += int(labels.numel())
    ce = total_nll / max(total_toks, 1)
    return {"ce": ce, "ppl": math.exp(min(ce, 50.0)), "tokens": total_toks}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-0 LRC-MoE Top-K' Expert Compression")
    p.add_argument("--model", required=True, help="Local path or HF id of teacher MoE model")
    p.add_argument("--out", required=True, help="Output directory for compressed native HF MoE")
    p.add_argument("--target-experts", type=int, required=True, help="Target number of experts K'")
    p.add_argument("--target-top-k", type=int, default=None, help="Optional target active top-m' per token")
    p.add_argument(
        "--method",
        default="lrc_cluster",
        choices=["freq", "energy", "gram_greedy", "lrc_cluster"],
    )
    p.add_argument("--router-mode", default="slice", choices=["slice", "kl_fit"])
    p.add_argument("--lrc-ridge", dest="use_lrc_ridge", action="store_true", default=True)
    p.add_argument("--no-lrc-ridge", dest="use_lrc_ridge", action="store_false")
    p.add_argument("--lrc-rank", type=int, default=0, help="0=full damped ridge; >0=low-rank SVD delta")
    p.add_argument("--sequential", dest="sequential", action="store_true", default=True)
    p.add_argument("--parallel", dest="sequential", action="store_false")
    p.add_argument(
        "--calib-files",
        nargs="+",
        default=[
            os.path.join(ROOT, "data/fw_score4_200M.jsonl"),
            os.path.join(ROOT, "data/pt_cosmopedia_v3_pool.jsonl"),
        ],
    )
    p.add_argument(
        "--holdout-file",
        default=os.path.join(ROOT, "data/fineweb_edu_holdout.jsonl"),
    )
    p.add_argument("--calib-seqs", type=int, default=64)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--calib-bs", type=int, default=2)
    p.add_argument("--probe-tokens", type=int, default=512)
    p.add_argument("--ridge-sweeps", type=int, default=3)
    p.add_argument("--ridge-alphas", type=str, default="0.2,1.0,5.0,20.0,80.0,250.0")
    p.add_argument("--router-steps", type=int, default=120)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-ppl", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dt_map[args.dtype]
    device = torch.device(args.device)

    print(f"[compress_moe_lrc] loading {args.model} ({args.dtype} -> {device}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kw = {"dtype": dtype}
    if getattr(torch.version, "hip", None):
        load_kw["experts_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    model.eval()

    calib_batches = load_calib_batches(
        tokenizer,
        args.calib_files,
        num_seqs=args.calib_seqs,
        seq_len=args.calib_seq_len,
        batch_size=args.calib_bs,
    )
    print(
        f"[compress_moe_lrc] calibration ready: {args.calib_seqs} seqs x {args.calib_seq_len} = "
        f"{args.calib_seqs * args.calib_seq_len:,} tokens",
        flush=True,
    )

    ppl_before = None
    if not args.skip_ppl and os.path.isfile(args.holdout_file):
        ppl_before = eval_heldout_ce(model, tokenizer, args.holdout_file, device=device)
        print(
            f"[compress_moe_lrc] teacher holdout CE={ppl_before['ce']:.4f} "
            f"PPL={ppl_before['ppl']:.2f} ({ppl_before['tokens']} tok)",
            flush=True,
        )

    alphas = tuple(float(x.strip()) for x in args.ridge_alphas.split(",") if x.strip())
    cfg = MoeCompressConfig(
        target_experts=args.target_experts,
        target_top_k=args.target_top_k,
        selection_method=args.method,
        router_mode=args.router_mode,
        use_lrc_ridge=args.use_lrc_ridge,
        lrc_rank=args.lrc_rank,
        sequential=args.sequential,
        probe_tokens=args.probe_tokens,
        ridge_sweeps=args.ridge_sweeps,
        ridge_alphas=alphas,
        router_steps=args.router_steps,
    )

    report = compress_moe_model(
        model=model,
        calib_batches=calib_batches,
        cfg=cfg,
        device=device,
        save_dir=args.out,
        tokenizer=tokenizer,
        verbose=True,
    )

    ppl_after = None
    if not args.skip_ppl and os.path.isfile(args.holdout_file):
        ppl_after = eval_heldout_ce(model, tokenizer, args.holdout_file, device=device)
        print(
            f"[compress_moe_lrc] compressed holdout CE={ppl_after['ce']:.4f} "
            f"PPL={ppl_after['ppl']:.2f} ({ppl_after['tokens']} tok)",
            flush=True,
        )

    report["holdout_before"] = ppl_before
    report["holdout_after"] = ppl_after
    with open(os.path.join(args.out, "lrc_compression_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"[compress_moe_lrc] DONE -> {args.out} | "
        f"params {report['params_before']/1e9:.2f}B -> {report['params_after']/1e9:.2f}B "
        f"({report['param_ratio']*100:.1f}%) | "
        f"mean valMSE {report['mean_init_val_rel_mse']:.4f} -> {report['mean_final_val_rel_mse']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
