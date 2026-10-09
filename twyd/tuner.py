"""ms-swift Tuner plugins: DenseLRC / CoreLRC / PlainLRC (part 1: CoreLinear)."""
from typing import Optional

import torch
from torch import nn


class CoreLinear(nn.Module):
    """Mergeable LRC linear: W = T Z_col^T (+ U_perp Z_perp^T), gate/up LEFT.

    Down (RIGHT): W = Zr @ R + Z_perp_r @ V_perp, shape r x d_ff.
    Z_perp=0  =>  exactly the plain-LRC student at step 0 (Prop.1).
    """

    def __init__(self, T_or_R: torch.Tensor, Z_col: torch.Tensor,
                 U_perp_or_V: Optional[torch.Tensor] = None,
                 Z_perp: Optional[torch.Tensor] = None,
                 side: str = "left", with_perp: bool = True):
        super().__init__()
        self.side = side
        self.with_perp = with_perp
        self.register_buffer("basis_col", T_or_R.detach().contiguous())
        self.z_col = nn.Parameter(Z_col.detach().contiguous())
        if with_perp:
            assert U_perp_or_V is not None and Z_perp is not None
            self.register_buffer("basis_perp", U_perp_or_V.detach().contiguous())
            self.z_perp = nn.Parameter(Z_perp.detach().contiguous())
        else:
            self.register_buffer("basis_perp", torch.empty(0))
            self.z_perp = None

    def merged_weight(self) -> torch.Tensor:
        if self.side == "left":
            W = self.basis_col @ self.z_col.T
            if self.with_perp:
                W = W + self.basis_perp @ self.z_perp.T
            return W
        W = self.z_col @ self.basis_col
        if self.with_perp:
            W = W + self.z_perp @ self.basis_perp
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.merged_weight())


from copy import deepcopy
from typing import Dict, List

from .core_basis import build_core_basis


def _mlp_module_names(model: nn.Module, target_modules: List[str]) -> List[str]:
    names = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        short = n.split(".")[-1]
        if short in target_modules or n in target_modules:
            names.append(n)
    return sorted(set(names))


def _get_parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


class _TwydWrapper(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, cfg):
        super().__init__()
        self.student = student
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.twyd_cfg = cfg
        self.config = getattr(student, "config", None)

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    def get_teacher_outputs(self, **inputs):
        with torch.no_grad():
            return self.teacher(**inputs)

    def gradient_checkpointing_enable(self, *a, **k):
        fn = getattr(self.student, "gradient_checkpointing_enable", None)
        if fn is not None:
            return fn(*a, **k)

    def save_pretrained(self, *a, **k):
        return self.student.save_pretrained(*a, **k)


def _build_factorized_student(student: nn.Module, teacher: nn.Module, cfg,
                              with_perp: bool) -> Dict[str, dict]:
    teacher_hidden = int(teacher.config.hidden_size)
    r = int(cfg.student_hidden) if cfg.student_hidden else teacher_hidden // 2
    meta: Dict[str, dict] = {}
    for mod_name in _mlp_module_names(student, list(cfg.target_modules)):
        parent, attr = _get_parent(student, mod_name)
        lin: nn.Linear = getattr(parent, attr)
        try:
            t_parent, _ = _get_parent(teacher, mod_name)
            t_weight = getattr(t_parent, attr).weight.detach()
        except Exception:
            continue
        short = mod_name.split(".")[-1]
        is_down = (short == "down_proj" or "down" in short or short in ("w2", "c_proj"))
        with torch.no_grad():
            if not is_down:
                d_ff, H = t_weight.shape
                side = "left"
                T, P = build_core_basis(t_weight, "left")
                s_w = lin.weight.detach().to(torch.float32)
                if s_w.shape != (d_ff, r):
                    tmp = torch.zeros(d_ff, r)
                    tmp[:, :min(r, s_w.shape[1])] = s_w[:d_ff, :min(r, s_w.shape[1])]
                    s_w = tmp
                Z_col = (T.T @ s_w).T.contiguous()
                Z_perp = torch.zeros(r, d_ff - H) if with_perp else None
                new_mod = CoreLinear(T, Z_col, P if with_perp else None,
                                     Z_perp, side="left", with_perp=with_perp)
            else:
                H, d_ff = t_weight.shape
                side = "right"
                R, Vp = build_core_basis(t_weight, "right")
                s_w = lin.weight.detach().to(torch.float32)
                if s_w.shape != (r, d_ff):
                    tmp = torch.zeros(r, d_ff)
                    tmp[:min(r, s_w.shape[0]), :] = s_w[:min(r, s_w.shape[0]), :d_ff]
                    s_w = tmp
                Z_col = (s_w @ R.T).contiguous()
                Z_perp = torch.zeros(r, d_ff - H) if with_perp else None
                new_mod = CoreLinear(R, Z_col, Vp if with_perp else None,
                                     Z_perp, side="right", with_perp=with_perp)
        setattr(parent, attr, new_mod.to(dtype=lin.weight.dtype))
        meta[mod_name] = {"side": side, "d_ff": d_ff, "H": H, "r": r,
                          "with_perp": with_perp, "branch": cfg.branch}
    return meta

def _freeze_for_branch(wrapper, cfg) -> None:
    if cfg.branch == "dense":
        wrapper.student.requires_grad_(True)
    else:
        wrapper.student.requires_grad_(False)
        for n, p in wrapper.student.named_parameters():
            if "z_col" in n or "z_perp" in n:
                p.requires_grad_(True)
    for p in wrapper.teacher.parameters():
        p.requires_grad_(False)


def _prepare_model_impl(args, model, branch: str):
    from .config import TwydLRCConfig as Cfg
    kw = dict(getattr(args, "tuner_kwargs", None) or {})
    cfg = Cfg(branch=branch,
              teacher_model=kw.get("teacher_model"),
              student_hidden=kw.get("student_hidden"),
              target_modules=kw.get("target_modules", ["gate_proj", "up_proj", "down_proj"]),
              kl_temperature=kw.get("kl_temperature", 40.0),
              kl_weight=kw.get("kl_weight", 1.0),
              ntp_weight=kw.get("ntp_weight", 1.0),
              aux_weight=kw.get("aux_weight", 0.2),
              use_aux=kw.get("use_aux", True),
              aux_loss_type=kw.get("aux_loss_type", "mseloss"),
              ridge_lambda=kw.get("ridge_lambda"),
              outlier_keep=kw.get("outlier_keep", 0))
    cfg.validate()
    teacher_id = cfg.teacher_model or getattr(args, "model", None)
    if teacher_id is None:
        raise ValueError("twyd needs tuner_kwargs.teacher_model")
    student = model
    try:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_id, torch_dtype=getattr(student, "dtype", torch.bfloat16),
            trust_remote_code=True)
    except Exception:
        teacher = deepcopy(student)
    teacher.eval()
    meta = _build_factorized_student(student, teacher, cfg, with_perp=(branch == "core"))
    wrapper = _TwydWrapper(student, teacher, cfg)
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)
    return wrapper


class DenseLRCAdapter:
    branch = "dense"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "dense")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        return model


class CoreLRCAdapter:
    branch = "core"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "core")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        return model


class PlainLRCAdapter:
    branch = "plain"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "plain")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        return model

