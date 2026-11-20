"""Phase 2 TWYD-MoE tuner: r=1536 Shared-Dense Mother + Fine-Grained Routed Experts.

Preserves hidden_size=r and KV-cache footprint identically to the 1.6B dense
mother, while expanding mid-to-late MLP knowledge storage via sparse Top-K
routed residual experts initialized to bit-exact step-0 identity (w_down=0).

Exported checkpoints are 100% self-contained HF directories loadable via:
    AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True)
and directly evaluable with scripts/run_lm_eval.sh.
"""
from __future__ import annotations

import json
import os
import shutil
import torch
from safetensors.torch import save_file

from .config import TwydLRCConfig, parse_layer_range
from .hf_moe.modeling_twyd_moe import TwydSharedRoutedMoE, inject_twyd_moe_
from .tuner import (
    _TunerBase,
    _TwydWrapper,
    _apply_pt_ckpt_,
    _attn_impl_of,
    _hidden_of,
    _load_pt_merged_,
    _merge_state_dict_,
    _pt_ckpt_action,
    _resolve_teacher,
    _shrink_student_,
    _twyd_kw,
)


def _moe_env():
    def _int(k, d):
        v = os.environ.get(k, "")
        return int(v) if str(v).strip() != "" else d

    def _float(k, d):
        v = os.environ.get(k, "")
        return float(v) if str(v).strip() != "" else d

    def _bool(k, d):
        v = os.environ.get(k, "")
        return (str(v).strip().lower() not in ("0", "false", "no", "off")) if str(v).strip() != "" else d

    return {
        "moe_layers": os.environ.get("TWYD_MOE_LAYERS", "8-23"),
        "num_experts": _int("TWYD_MOE_EXPERTS", 8),
        "top_k": _int("TWYD_MOE_TOPK", 2),
        "moe_intermediate_size": _int("TWYD_MOE_INTER", 1536),
        "lb_coef": _float("TWYD_MOE_LB_COEF", 0.02),
        "z_coef": _float("TWYD_MOE_Z_COEF", 0.001),
        "unfreeze_shared": _bool("TWYD_MOE_UNFREEZE_SHARED", False),
        "router_lr_scale": _float("TWYD_MOE_ROUTER_LR_SCALE", 2.0),
    }


def _freeze_for_moe(wrapper: _TwydWrapper, cfg: TwydLRCConfig, moe_cfg: dict) -> None:
    """Freeze the 1.6B dense mother; train routed knowledge experts (+ optional shared)."""
    unfreeze_shared = bool(moe_cfg.get("unfreeze_shared", False))
    writer_grad_scale = float(getattr(cfg, "writer_grad_scale", 1.0) or 1.0)
    writer_max_norm = float(getattr(cfg, "writer_max_norm", 0.0) or 0.0)

    wrapper.student.requires_grad_(False)
    for n, p in wrapper.student.named_parameters():
        lname = n.lower()
        is_routed = any(k in lname for k in (".router.", ".w_gate", ".w_up", ".w_down"))
        if is_routed:
            p.requires_grad_(True)
        elif unfreeze_shared and ".shared_mlp." in lname:
            p.requires_grad_(True)

        # Optional per-tensor gradient cap/scale on routed w_down so experts write
        # smoothly into the 1536-d residual bus without destabilizing step sizes.
        if p.requires_grad and ".w_down" in lname:
            if (abs(writer_grad_scale - 1.0) > 1e-6 or writer_max_norm > 0.0) and not getattr(p, "_twyd_writer_hooked", False):
                _scale = writer_grad_scale
                _mmax = writer_max_norm

                def _hook(g, s=_scale, m=_mmax):
                    if g is None:
                        return None
                    out_g = g
                    if m > 0.0:
                        gn = out_g.float().norm()
                        if gn > m:
                            out_g = out_g * (m / (gn + 1e-6)).to(out_g.dtype)
                    if abs(s - 1.0) > 1e-6:
                        out_g = out_g * s
                    return out_g

                p.register_hook(_hook)
                p._twyd_writer_hooked = True

    for p in wrapper.teacher.parameters():
        p.requires_grad_(False)


def save_twyd_moe_pretrained(model, save_directory: str, safe_serialization: bool = True) -> str:
    """Export a self-contained HF TwydMoE checkpoint (weights + config + auto_map code)."""
    os.makedirs(save_directory, exist_ok=True)
    student = getattr(model, "student", model)
    moe_info = getattr(model, "twyd_moe_info", None) or {}

    # Save weights (clone tied storages because safetensors forbids shared ptrs).
    sd = {}
    seen = set()
    for n, p in student.named_parameters():
        t = p.detach().cpu()
        if t.numel() > 0:
            ptr = t.data_ptr()
            if ptr in seen:
                t = t.clone()
            else:
                seen.add(ptr)
        sd[n] = t
    save_file(sd, os.path.join(save_directory, "model.safetensors"))

    # Copy self-contained trust_remote_code HF modules into the checkpoint dir.
    hf_moe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_moe")
    for fname in ("configuration_twyd_moe.py", "modeling_twyd_moe.py"):
        src = os.path.join(hf_moe_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_directory, fname))

    # Write config.json with auto_map pointing to the local trust_remote_code modules.
    cfg_obj = getattr(student, "config", None)
    cfg_dict = json.loads(cfg_obj.to_json_string()) if cfg_obj is not None else {}
    cfg_dict["model_type"] = "twyd_moe"
    cfg_dict["architectures"] = ["TwydMoEForCausalLM"]
    cfg_dict["auto_map"] = {
        "AutoConfig": "configuration_twyd_moe.TwydMoEConfig",
        "AutoModelForCausalLM": "modeling_twyd_moe.TwydMoEForCausalLM",
    }
    cfg_dict["moe_layers"] = list(moe_info.get("moe_layers", []))
    cfg_dict["num_experts"] = int(moe_info.get("num_experts", 8))
    cfg_dict["num_experts_per_tok"] = int(moe_info.get("top_k", 2))
    cfg_dict["moe_intermediate_size"] = int(moe_info.get("moe_intermediate_size", 1536))
    cfg_dict["moe_lb_coef"] = float(moe_info.get("lb_coef", 0.02))
    cfg_dict["moe_z_coef"] = float(moe_info.get("z_coef", 0.001))
    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    gen_cfg = getattr(student, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.save_pretrained(save_directory)
        except Exception:
            pass
    return save_directory


def _detect_moe_ckpt_cfg(pt_ckpt: str | None) -> dict | None:
    """Return TwydMoE config dict if pt_ckpt is already an exported TwydMoE dir."""
    if not pt_ckpt or not os.path.isdir(str(pt_ckpt)):
        return None
    cfg_path = os.path.join(str(pt_ckpt), "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("model_type") == "twyd_moe" and d.get("moe_layers"):
            return d
    except Exception:
        return None
    return None


class MoeLRCAdapter(_TunerBase):
    branch = "moe"

    @staticmethod
    def prepare_model(args, model):
        kw = _twyd_kw(args)
        moe_cfg = _moe_env()
        cfg = TwydLRCConfig(
            branch="moe",
            teacher_model=kw.get("teacher_model"),
            student_hidden=kw.get("student_hidden"),
            stage=str(kw.get("stage", "pt")).lower(),
            warm_start=str(kw.get("warm_start", "base")).lower(),
            kl_temperature=float(kw.get("kl_temperature", 1.0)),
            kl_weight=float(kw.get("kl_weight", 5.0)),
            ntp_weight=float(kw.get("ntp_weight", 1.0)),
            aux_weight=float(kw.get("aux_weight", 0.0)),
            use_aux=bool(kw.get("use_aux", False)),
            aux_loss_type=str(kw.get("aux_loss_type", "mseloss") or "mseloss").lower(),
            aux_layers=str(kw.get("aux_layers", "") or ""),
            kl_topk=int(kw.get("kl_topk", 256) or 0),
            rho_keep=float(kw.get("rho_keep", 0.35) if kw.get("rho_keep") is not None else 0.35),
            rho_min_weight=float(kw.get("rho_min_weight", 0.1) if kw.get("rho_min_weight") is not None else 0.1),
            writer_lr_scale=float(kw.get("writer_lr_scale", 1.0) if kw.get("writer_lr_scale") is not None else 1.0),
            writer_grad_scale=float(kw.get("writer_grad_scale", 1.0) if kw.get("writer_grad_scale") is not None else 1.0),
            writer_max_norm=float(kw.get("writer_max_norm", 0.0) if kw.get("writer_max_norm") is not None else 0.0),
        )
        cfg.validate()

        teacher_id = cfg.teacher_model or getattr(args, "model", None)
        if teacher_id is None:
            raise ValueError("twyd moe_lrc needs TWYD_TEACHER")
        student = model
        attn_impl = _attn_impl_of(student)
        teacher = _resolve_teacher(student, teacher_id, attn_impl=attn_impl)
        teacher.eval()

        if not cfg.student_hidden:
            cfg.student_hidden = max(64, _hidden_of(teacher) // 2)
        r = int(cfg.student_hidden)
        if _hidden_of(student) != r:
            student = _shrink_student_(student, teacher, r, attn_impl=attn_impl)

        pt_ckpt = kw.get("pt_ckpt", None)
        load_ckpt = (_pt_ckpt_action(cfg, pt_ckpt) == "load")
        ckpt_moe_cfg = _detect_moe_ckpt_cfg(pt_ckpt) if load_ckpt else None

        n_layers = int(getattr(getattr(student, "config", None), "num_hidden_layers", 28) or 28)
        if ckpt_moe_cfg is not None:
            # Continuing / SFT'ing from an existing TwydMoE checkpoint:
            # inject matching MoE layout FIRST, then load all shared + routed weights!
            layer_list = sorted(int(i) for i in ckpt_moe_cfg["moe_layers"])
            moe_cfg["moe_layers"] = ",".join(str(i) for i in layer_list)
            moe_cfg["num_experts"] = int(ckpt_moe_cfg.get("num_experts", moe_cfg["num_experts"]))
            moe_cfg["top_k"] = int(ckpt_moe_cfg.get("num_experts_per_tok", moe_cfg["top_k"]))
            moe_cfg["moe_intermediate_size"] = int(
                ckpt_moe_cfg.get("moe_intermediate_size", moe_cfg["moe_intermediate_size"])
            )
            injected = inject_twyd_moe_(
                student,
                moe_layers=layer_list,
                num_experts=moe_cfg["num_experts"],
                top_k=moe_cfg["top_k"],
                moe_intermediate_size=moe_cfg["moe_intermediate_size"],
                lb_coef=moe_cfg["lb_coef"],
                z_coef=moe_cfg["z_coef"],
            )
            _apply_pt_ckpt_(student, pt_ckpt, cfg.stage,
                            strict=os.environ.get("TWYD_PT_CKPT_STRICT", "1") != "0")
            print(f"[twyd-moe] resumed existing TwydMoE ckpt from {pt_ckpt} "
                  f"(layers={len(injected)}, E={moe_cfg['num_experts']}, topK={moe_cfg['top_k']})",
                  flush=True)
        else:
            # Upcycling from a Dense mother checkpoint: load Dense weights first,
            # then wrap selected MLPs with zero-init routed experts (step-0 identity).
            if load_ckpt:
                _apply_pt_ckpt_(student, pt_ckpt, cfg.stage,
                                strict=os.environ.get("TWYD_PT_CKPT_STRICT", "1") != "0")
            layer_list = sorted(parse_layer_range(moe_cfg["moe_layers"], n_layers))
            injected = inject_twyd_moe_(
                student,
                moe_layers=layer_list,
                num_experts=moe_cfg["num_experts"],
                top_k=moe_cfg["top_k"],
                moe_intermediate_size=moe_cfg["moe_intermediate_size"],
                lb_coef=moe_cfg["lb_coef"],
                z_coef=moe_cfg["z_coef"],
            )
        gas = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
        for mod in student.modules():
            if isinstance(mod, TwydSharedRoutedMoE):
                mod.aux_grad_scale = 1.0 / gas
        wrapper = _TwydWrapper(student, teacher, cfg)
        wrapper.twyd_moe_info = {
            "moe_layers": injected,
            "num_experts": moe_cfg["num_experts"],
            "top_k": moe_cfg["top_k"],
            "moe_intermediate_size": moe_cfg["moe_intermediate_size"],
            "lb_coef": moe_cfg["lb_coef"],
            "z_coef": moe_cfg["z_coef"],
            "router_lr_scale": moe_cfg["router_lr_scale"],
        }

        # Give router a slightly faster LR if requested (default 2.0x) so routing specializes early.
        orig_param_groups = wrapper.create_optimizer_param_groups

        def _moe_param_groups(lr: float = 1e-4, weight_decay: float = 0.0, **kwargs):
            r_scale = float(moe_cfg.get("router_lr_scale", 1.0) or 1.0)
            w_scale = float(getattr(cfg, "writer_lr_scale", 1.0) or 1.0)
            if abs(r_scale - 1.0) < 1e-9 and abs(w_scale - 1.0) < 1e-9:
                return orig_param_groups(lr=lr, weight_decay=weight_decay, **kwargs)
            groups = {}
            for n, p in wrapper.student.named_parameters():
                if not p.requires_grad:
                    continue
                ln = n.lower()
                no_decay = ("bias" in ln or "norm" in ln or "router" in ln)
                wd = 0.0 if no_decay else float(weight_decay)
                if ".router." in ln:
                    p_lr = float(lr) * r_scale
                elif ".w_down" in ln or "down_proj" in ln:
                    p_lr = float(lr) * w_scale
                else:
                    p_lr = float(lr)
                key = (round(p_lr, 12), round(wd, 8))
                groups.setdefault(key, []).append(p)
            return [{"params": ps, "lr": k_lr, "weight_decay": k_wd}
                    for (k_lr, k_wd), ps in groups.items() if ps]

        wrapper.create_optimizer_param_groups = _moe_param_groups
        _freeze_for_moe(wrapper, cfg, moe_cfg)
        n_tot = sum(p.numel() for p in student.parameters()) / 1e6
        n_trn = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
        print(f"[twyd-moe] injected {len(injected)} MoE layers ({moe_cfg['moe_layers']}): "
              f"E={moe_cfg['num_experts']} topK={moe_cfg['top_k']} d_e={moe_cfg['moe_intermediate_size']} | "
              f"total={n_tot:.1f}M trainable={n_trn:.1f}M", flush=True)
        return wrapper

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        _merge_state_dict_(model, state_dict)
        save_twyd_moe_pretrained(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        if os.path.isdir(str(model_id)):
            student = getattr(model, "student", model)
            n = _load_pt_merged_(student, str(model_id))
            if n > 0:
                return model
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(model, "dtype", torch.bfloat16), trust_remote_code=True)
