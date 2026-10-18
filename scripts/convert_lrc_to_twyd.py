"""Convert a Low-Rank Clone (LRC) checkpoint into a twyd-initializable HF student.

LowRankClone stores per-MLP factors (gate_proj/up_proj/down_proj + zoom_gate/zoom_up/zoom_down).
TWYD's CoreLinear deploys W = T Z_col^T  (left/gate-up) or W = Z_col R  (right/down), where T/R are
the teacher SVD basis and Z_col is the trainable factor (r x H).

Conversion (per MLP module):
  gate/up (left):  W_lrc = gate_proj.weight (d_ff x r)  [or teacher slice if not saved]
                   T     = teacher SVD basis (d_ff x H)  [frozen, same as twyd]
                   Z_col = (T^T @ W_lrc).T              (r x H)  -> twyd trainable
  down   (right): W_lrc = down_proj.weight (r x d_ff)
                   R     = teacher down row-basis (H x d_ff) [frozen]
                   Z_col = W_lrc @ R.T                    (r x H) -> twyd trainable

The resulting twyd CoreLinear(T, Z_col, U_perp, Z_perp=0) is zero-init-equivalent to plain-LRC
(Prop.1) and can be loaded into twyd via --base + --teacher, then trained with --tuner_type
dense_lrc/core_lrc/plain_lrc --loss_type lrc_distill.

Usage:
  python scripts/convert_lrc_to_twyd.py \
      --lrc-ckpt <LowRankClone model.safetensors dir> \
      --teacher <teacher HF id/path> \
      --student-base <student base HF id/path> \
      --save <output twyd-init dir> \
      --branch core|dense|plain \
      --target-hidden 1536
"""
import argparse
import json
import os
import sys

_ROOT = os.environ.get("TWYD_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from safetensors.torch import load_file, save_file


def _load_sd(ckpt: str) -> dict:
    src = os.path.join(ckpt, "model.safetensors")
    if os.path.isfile(src):
        return load_file(src)
    raise FileNotFoundError(f"no model.safetensors in {ckpt}")


def _teacher_left_basis(t_weight: torch.Tensor):
    """T = U[:, :H] for gate/up (d_ff x H, d_ff > H)."""
    U, _, _ = torch.linalg.svd(t_weight.detach().to(torch.float32).cpu(), full_matrices=True)
    d_ff, H = t_weight.shape
    return U[:, :H].contiguous(), U[:, H:].contiguous()


def _teacher_right_basis(t_weight: torch.Tensor):
    """R = Vh[:H, :] for down (H x d_ff row-space basis), V_perp = Vh[H:, :]."""
    _, _, Vh = torch.linalg.svd(t_weight.detach().to(torch.float32).cpu(), full_matrices=True)
    H, d_ff = t_weight.shape
    return Vh[:H, :].contiguous(), Vh[H:, :].contiguous()


def _slice_teacher_weight(t_weight: torch.Tensor, r: int, is_down: bool) -> torch.Tensor:
    """Warm-start W_lrc by slicing the teacher weight to width r (LowRankClone style)."""
    t_w = t_weight.detach().to(torch.float32).cpu()
    if is_down:
        return t_w[:r, :].contiguous()
    else:
        return t_w[:, :r].contiguous()


def convert(lrc_ckpt: str, teacher_id: str, student_base: str, save_dir: str,
            branch: str, r: int):
    from transformers import AutoModelForCausalLM, AutoConfig

    os.makedirs(save_dir, exist_ok=True)
    lrc_sd = _load_sd(lrc_ckpt)

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
    ).eval()
    student = AutoModelForCausalLM.from_pretrained(
        student_base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    tc = getattr(teacher.config, "text_config", None) or teacher.config
    teacher_H = int(tc.hidden_size)
    teacher_dff = int(getattr(tc, "intermediate_size", 0) or 0)

    out_sd = {}
    meta = {}
    converted = 0

    mlp_factor_keys = ("gate_proj", "up_proj", "down_proj", "zoom_gate", "zoom_up", "zoom_down")
    for k, v in lrc_sd.items():
        short = k.split("model.")[-1] if k.startswith("model.") else k
        is_mlp_factor = any(f".{m}." in short or short.endswith(f".{m}") for m in mlp_factor_keys)
        if not is_mlp_factor:
            out_sd[k] = v.detach().cpu() if torch.is_tensor(v) else v

    num_layers = int(getattr(tc, "num_hidden_layers", 0))
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}.mlp."
        for short_name, is_down in (("gate_proj", False), ("up_proj", False), ("down_proj", True)):
            w_key = prefix + short_name + ".weight"
            t_cand = None
            for n, p in teacher.named_parameters():
                if n.endswith(short_name + ".weight") and f"layers.{layer_idx}.mlp." in n:
                    t_cand = p
                    break
            if t_cand is None:
                continue
            t_w = t_cand.detach().to(torch.float32).cpu()

            if w_key in lrc_sd:
                W_lrc = lrc_sd[w_key].detach().to(torch.float32).cpu()
            else:
                W_lrc = _slice_teacher_weight(t_w, r, is_down)

            expected = (r, teacher_dff) if is_down else (teacher_dff, r)
            if tuple(W_lrc.shape) != tuple(expected):
                print(f"[lrc->twyd] skip {w_key}: shape {tuple(W_lrc.shape)} != {expected}")
                continue

            if not is_down:
                T, U_perp = _teacher_left_basis(t_w)
                Z_col = (T.T @ W_lrc).T.contiguous()
                Z_perp = torch.zeros(r, teacher_dff - teacher_H)
                out_sd[prefix + short_name + ".z_col"] = Z_col
                out_sd[prefix + short_name + ".z_perp"] = Z_perp
                out_sd[prefix + short_name + ".basis_col"] = T
                out_sd[prefix + short_name + ".basis_perp"] = U_perp
                out_sd.pop(w_key, None)
            else:
                R, V_perp = _teacher_right_basis(t_w)
                Z_col = (W_lrc @ R.T).contiguous()
                Z_perp = torch.zeros(r, teacher_dff - teacher_H)
                out_sd[prefix + short_name + ".z_col"] = Z_col
                out_sd[prefix + short_name + ".z_perp"] = Z_perp
                out_sd[prefix + short_name + ".basis_col"] = R
                out_sd[prefix + short_name + ".basis_perp"] = V_perp
                out_sd.pop(w_key, None)

            converted += 1
            meta[w_key] = {"r": r, "H": teacher_H, "d_ff": teacher_dff,
                           "down": is_down, "branch": branch}

    save_file(out_sd, os.path.join(save_dir, "model.safetensors"))

    try:
        cfg = AutoConfig.from_pretrained(student_base, trust_remote_code=True)
        tcfg = getattr(cfg, "text_config", None) or cfg
        tcfg.hidden_size = int(r)
        head_dim = int(getattr(tcfg, "head_dim", None) or (int(tcfg.hidden_size) // int(tcfg.num_attention_heads)))
        if head_dim > 0 and r % head_dim == 0:
            tcfg.num_attention_heads = r // head_dim
        if getattr(tcfg, "num_key_value_heads", None):
            tcfg.num_key_value_heads = max(1, min(int(tcfg.num_key_value_heads), r // head_dim))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(json.loads(cfg.to_json_string()), f, indent=2)
    except Exception as e:
        print(f"[lrc->twyd] config update skipped: {e}")

    with open(os.path.join(save_dir, "lrc_to_twyd_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[lrc->twyd] converted {converted} MLP modules -> {save_dir}")
    print(f"[lrc->twyd] r={r}, teacher_H={teacher_H}, teacher_dff={teacher_dff}, branch={branch}")
    print("[lrc->twyd] Load with: --base <save_dir> --teacher <teacher> --tuner_type core_lrc/dense_lrc")


def main():
    ap = argparse.ArgumentParser(description="Convert LowRankClone checkpoint to twyd init")
    ap.add_argument("--lrc-ckpt", required=True, help="LowRankClone model.safetensors dir")
    ap.add_argument("--teacher", required=True, help="teacher HF id/path (for SVD basis)")
    ap.add_argument("--student-base", required=True, help="student base HF id/path (architecture)")
    ap.add_argument("--save", required=True, help="output twyd-init dir")
    ap.add_argument("--branch", default="core", choices=["core", "dense", "plain"])
    ap.add_argument("--target-hidden", type=int, default=1536, help="r = student hidden width")
    args = ap.parse_args()
    convert(args.lrc_ckpt, args.teacher, args.student_base, args.save,
            args.branch, args.target_hidden)


if __name__ == "__main__":
    main()

