"""ms-swift Tuner plugins: DenseLRC / CoreLRC / PlainLRC (part 1: CoreLinear)."""
import os
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


class DenseLinearLoRA(nn.Module):
    """LoRA-wrapped dense linear for Train-What-You-Deploy.

    W = base_weight(frozen) + (lora_A @ lora_B) * (alpha / rank)

    base_weight is a buffer (no grad, no optimizer state); only lora_A / lora_B
    are trainable. merged_weight() returns the full deployed W for export, so
    inference stays a single dense matmul (zero extra latency).
    """
    def __init__(self, in_features: int, out_features: int,
                 lora_rank: int = 64, lora_alpha: float = 16.0,
                 base_weight: Optional[torch.Tensor] = None, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = int(lora_rank)
        self.scaling = float(lora_alpha) / self.lora_rank if self.lora_rank > 0 else 0.0

        if base_weight is not None:
            self.register_buffer("base_weight", base_weight.detach().contiguous())
        else:
            self.register_buffer("base_weight", torch.empty(out_features, in_features))

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.lora_B = nn.Parameter(torch.empty(self.lora_rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def merged_weight(self) -> torch.Tensor:
        W = self.base_weight
        if self.lora_rank > 0:
            W = W + (self.lora_A @ self.lora_B) * self.scaling
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.merged_weight())


from copy import deepcopy
import math
from typing import Dict, List

from .core_basis import build_core_basis


try:
    from swift.tuner_plugin.base import Tuner as _TunerBase
except Exception:  # standalone test without ms-swift
    _TunerBase = object


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
    """Thin ms-swift-compatible wrapper: delegates HF API to student.

    ms-swift trainer mixin checks isinstance(model, (SwiftModel, PeftModel))
    and reads model.model_meta / model.model_dir / generate() etc. We expose
    those attributes by delegation so the stock `swift pt/sft` loop works
    without forking ms-swift. Teacher stays a frozen sibling (never saved).
    """
    def __init__(self, student: nn.Module, teacher: nn.Module, cfg):
        super().__init__()
        self.student = student
        # Teacher must NOT be a registered submodule: DataParallel.replicate
        # walks _parameters/_modules and chokes on the duplicated (tied /
        # shared-storage) teacher tensors. Keep it as a plain attribute.
        object.__setattr__(self, "teacher", teacher)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.twyd_cfg = cfg
        self.config = getattr(student, "config", None)

    def state_dict(self, *args, **kwargs):
        sd = self.student.state_dict(*args, **kwargs)
        # Tied embeddings (Llama-3.2 tie_word_embeddings=true) put both
        # `model.embed_tokens.weight` and `lm_head.weight` on one storage;
        # the raw safetensors save_file path swift uses for plain wrappers
        # rejects shared storages. Clone duplicates: the values are
        # identical, and from_pretrained re-ties them harmlessly.
        seen = set()
        for k in list(sd.keys()):
            v = sd[k]
            if not torch.is_tensor(v) or v.numel() == 0:
                continue
            ptr = v.data_ptr()
            if ptr in seen:
                sd[k] = v.detach().clone()
            else:
                seen.add(ptr)
        return sd

    # --- ms-swift / HF attribute delegation (read-only -> student) ---
    def __getattr__(self, name: str):
        # 1) nn.Module submodule/param/buffer lookup (student/teacher live in
        # _modules, NOT __dict__; our override shadows nn.Module.__getattr__).
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # 2) dunder / init-time guard: raise cleanly instead of recursing.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            student = nn.Module.__getattr__(self, "student")
        except AttributeError:
            raise AttributeError(name)
        passthrough = (
            "model_meta", "model_dir", "model_info", "model_type",
            "generation_config", "origin_generation_config",
            "dtype", "name_or_path",
        )
        if name == "hf_device_map":
            # accelerate.verify_device_map does len(m.hf_device_map); empty
            # dict = single-device, no dispatch. Never return None.
            try:
                dm = getattr(student, "hf_device_map", None)
                if isinstance(dm, dict) and dm:
                    return dm
            except Exception:
                pass
            return {}
        if name in passthrough:
            return getattr(student, name, None)
        if name in ("base_model", "model"):
            return student
        raise AttributeError(f"_TwydWrapper has no attribute {name!r}")

    def generate(self, *args, **kwargs):
        return self.student.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        fn = getattr(self.student, "gradient_checkpointing_enable", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def get_input_embeddings(self):
        return self.student.get_input_embeddings()

    def get_output_embeddings(self):
        return self.student.get_output_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.student.resize_token_embeddings(*args, **kwargs)

    # --- HF passthroughs ms-swift touches on the model object ---
    @property
    def device(self):
        try:
            return next(self.student.parameters()).device
        except Exception:
            return None

    @property
    def name_or_path(self):
        try:
            return self.student.config.name_or_path
        except Exception:
            return None

    def tie_weights(self, *args, **kwargs):
        fn = getattr(self.student, "tie_weights", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        fn = getattr(self.student, "gradient_checkpointing_disable", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def enable_input_require_grads(self, *args, **kwargs):
        fn = getattr(self.student, "enable_input_require_grads", None)
        if fn is not None:
            return fn(*args, **kwargs)

    def disable_input_require_grads(self, *args, **kwargs):
        fn = getattr(self.student, "disable_input_require_grads", None)
        if fn is not None:
            return fn(*args, **kwargs)

    @property
    def is_gradient_checkpointing(self):
        try:
            return bool(getattr(self.student, "is_gradient_checkpointing", False))
        except Exception:
            return False

    def state_dict(self, *args, **kwargs):
        # Only student params flow to the trainer's checkpoint logic;
        # our Tuner.save_pretrained merges CoreLinear factors anyway.
        return self.student.state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.student.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.student.parameters(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # Cache last student inputs so LrcDistillLoss can run the frozen
        # teacher on the same batch (ms-swift strips inputs before loss_fn).
        # NOTE: positional input_ids (swift may pass input_ids positionally).
        try:
            keep = {}
            if args and isinstance(args[0], torch.Tensor) and "input_ids" not in kwargs:
                keep["input_ids"] = args[0].detach()
            for k in ("input_ids", "attention_mask", "position_ids"):
                if k in kwargs and kwargs[k] is not None and k not in keep:
                    v = kwargs[k]
                    keep[k] = v.detach() if isinstance(v, torch.Tensor) else v
            if keep:
                self._twyd_last_inputs = keep
        except Exception:
            pass
        return self.student(*args, **kwargs)

    def get_teacher_outputs(self, **inputs):
        t = self.teacher
        # Teacher may live on CPU (see _resolve_teacher); run it on the
        # student's device WITHOUT mutating/copying the stored module.
        # torch.func.functional_call runs the same graph with mapped params.
        try:
            from torch.func import functional_call as _fcall
        except Exception:
            _fcall = None
        try:
            ids = inputs.get("input_ids", None)
            if isinstance(ids, torch.Tensor):
                tdev = next(t.parameters()).device
                if tdev != ids.device and _fcall is not None:
                    tparams = {k: v.to(ids.device) for k, v in t.named_parameters()}
                    tbuffers = {k: v.to(ids.device) for k, v in t.named_buffers()}
                    inputs = {k: (v.to(ids.device) if isinstance(v, torch.Tensor) else v)
                              for k, v in inputs.items()}
                    try:
                        sdt = next(self.student.parameters()).dtype
                        tparams = {k: (v.to(sdt) if v.is_floating_point() else v)
                                   for k, v in tparams.items()}
                    except Exception:
                        pass
                    with torch.no_grad():
                        return _fcall(t, (tparams, tbuffers), tuple(), inputs)
                if tdev != ids.device and _fcall is None:
                    # Fallback (old torch): shallow copy is cheaper than deepcopy.
                    import copy as _copy
                    t = _copy.copy(t).to(ids.device)
                    try:
                        sdt = next(self.student.parameters()).dtype
                        t = t.to(sdt)
                    except Exception:
                        pass
                    inputs = {k: (v.to(ids.device) if isinstance(v, torch.Tensor) else v)
                              for k, v in inputs.items()}
            else:
                try:
                    sdev = next(self.student.parameters()).device
                    tdev = next(t.parameters()).device
                    if tdev != sdev and _fcall is None:
                        import copy as _copy
                        t = _copy.copy(t).to(sdev)
                except Exception:
                    pass
        except Exception:
            pass
        # Never ask teacher for hidden states unless loss needs them; keep the
        # default off to save memory (loss enables it per-batch if use_aux).
        with torch.no_grad():
            return t(**inputs)

    def save_pretrained(self, *a, **k):
        return self.student.save_pretrained(*a, **k)


def _load_pt_merged_(student: nn.Module, pt_ckpt: str) -> int:
    """Load a merged PT checkpoint into the student before factorized rebuild.

    Returns number of tensors copied. Missing/extra keys are ignored so a
    slightly different width config still boots (slice fallback in builder).
    """
    import os as _os
    n = 0
    sd = None
    try:
        from safetensors.torch import load_file as _load
        for cand in (str(pt_ckpt), _os.path.join(str(pt_ckpt), "model.safetensors")):
            if _os.path.isfile(cand):
                sd = _load(cand)
                break
    except Exception:
        sd = None
    if sd is None:
        try:
            import torch as _t
            for cand in (str(pt_ckpt), _os.path.join(str(pt_ckpt), "pytorch_model.bin")):
                if _os.path.isfile(cand):
                    sd = _t.load(cand, map_location="cpu", weights_only=True)
                    break
        except Exception:
            sd = None
    if sd is None:
        return 0
    with torch.no_grad():
        own = dict(student.named_parameters())
        for k, v in sd.items():
            short = k.split("student.")[-1] if "student." in k else k
            for cand_key in (k, short):
                if cand_key in own and tuple(own[cand_key].shape) == tuple(v.shape):
                    own[cand_key].copy_(v.to(own[cand_key].dtype))
                    n += 1
                    break
    return n


def _build_factorized_student(student: nn.Module, teacher: nn.Module, cfg,
                              branch: str) -> Dict[str, dict]:
    """Build TWYD student MLP.

    plain: CoreLinear(T, Z_col, with_perp=False) -> r x H trainable.
    core:  CoreLinear(T, Z_col, U_perp, Z_perp=0, with_perp=True) -> r x d_ff trainable (Prop.1).
    dense: nn.Linear(d_ff x r, trainable) init at merged plain weight (paper Sec 4.1).
    """
    with_perp = (branch == "core")
    is_dense = (branch == "dense")
    lora_rank = int(getattr(cfg, "lora_rank", 0) or 0)
    lora_alpha = float(getattr(cfg, "lora_alpha", 16.0) or 16.0)
    tcfg = getattr(teacher, "config", None)
    teacher_hidden = int(getattr(tcfg, "hidden_size", getattr(getattr(tcfg, "text_config", None), "hidden_size", 0)) or 0)
    if teacher_hidden == 0:
        teacher_hidden = int(getattr(getattr(student, "config", None), "hidden_size", 0) or 0) * 2
    r = int(cfg.student_hidden) if cfg.student_hidden else teacher_hidden // 2
    meta: Dict[str, dict] = {}
    for mod_name in _mlp_module_names(student, list(cfg.target_modules)):
        parent, attr = _get_parent(student, mod_name)
        lin: nn.Linear = getattr(parent, attr)
        try:
            t_parent, _ = _get_parent(teacher, mod_name)
            t_weight = getattr(t_parent, attr).weight.detach().cpu()
        except Exception:
            continue
        short = mod_name.split(".")[-1]
        is_down = (short == "down_proj" or "down" in short or short in ("w2", "c_proj"))
        lin_dev, lin_dt = lin.weight.device, lin.weight.dtype
        with torch.no_grad():
            if not is_down:
                d_ff, H = t_weight.shape
                side = "left"
                T, P = build_core_basis(t_weight, "left")
                s_w = lin.weight.detach().to(torch.float32).cpu()
                if s_w.shape != (d_ff, r):
                    tmp = torch.zeros(d_ff, r)
                    tmp[:, :min(r, s_w.shape[1])] = s_w[:d_ff, :min(r, s_w.shape[1])]
                    s_w = tmp
                Z_col = (T.T @ s_w).T.contiguous()
                W0 = (T @ Z_col.T).contiguous()  # merged plain weight d_ff x r
                if is_dense:
                    if lora_rank > 0:
                        new_mod = DenseLinearLoRA(r, d_ff, lora_rank=lora_rank,
                                                   lora_alpha=lora_alpha,
                                                   base_weight=W0)
                    else:
                        new_mod = nn.Linear(r, d_ff, bias=False)
                        with torch.no_grad():
                            new_mod.weight.copy_(W0.to(new_mod.weight.dtype))
                else:
                    Z_perp = torch.zeros(r, d_ff - H) if with_perp else None
                    new_mod = CoreLinear(T, Z_col, P if with_perp else None,
                                         Z_perp, side="left", with_perp=with_perp)
            else:
                H, d_ff = t_weight.shape
                side = "right"
                R, Vp = build_core_basis(t_weight, "right")
                s_w = lin.weight.detach().to(torch.float32).cpu()
                if s_w.shape != (r, d_ff):
                    tmp = torch.zeros(r, d_ff)
                    tmp[:min(r, s_w.shape[0]), :] = s_w[:min(r, s_w.shape[0]), :d_ff]
                    s_w = tmp
                Z_col = (s_w @ R.T).contiguous()
                W0 = (Z_col @ R).contiguous()  # merged plain weight r x d_ff
                if is_dense:
                    if lora_rank > 0:
                        new_mod = DenseLinearLoRA(d_ff, r, lora_rank=lora_rank,
                                                   lora_alpha=lora_alpha,
                                                   base_weight=W0)
                    else:
                        new_mod = nn.Linear(d_ff, r, bias=False)
                        with torch.no_grad():
                            new_mod.weight.copy_(W0.to(new_mod.weight.dtype))
                else:
                    Z_perp = torch.zeros(r, d_ff - H) if with_perp else None
                    new_mod = CoreLinear(R, Z_col, Vp if with_perp else None,
                                         Z_perp, side="right", with_perp=with_perp)
        # basis algebra ran on CPU fp32; move the rebuilt module back onto
        # the device/dtype of the linear it replaces.
        setattr(parent, attr, new_mod.to(device=lin_dev, dtype=lin_dt))
        meta[mod_name] = {"side": side, "d_ff": d_ff, "H": H, "r": r,
                          "with_perp": with_perp, "is_dense": is_dense,
                          "branch": cfg.branch, "lora_rank": lora_rank,
                          "lora_alpha": lora_alpha}
    return meta


def _freeze_for_branch(wrapper, cfg) -> None:
    if cfg.branch == "dense":
        # Train-what-you-deploy: only MLP linears we replaced stay trainable;
        # keep attention/norm/embed frozen (paper App.B, minimal change).
        wrapper.student.requires_grad_(False)
        for n, p in wrapper.student.named_parameters():
            if any(k in n for k in list(cfg.target_modules or [])):
                p.requires_grad_(True)
    else:
        wrapper.student.requires_grad_(False)
        for n, p in wrapper.student.named_parameters():
            if "z_col" in n or "z_perp" in n:
                p.requires_grad_(True)
    for p in wrapper.teacher.parameters():
        p.requires_grad_(False)


def _merge_state_dict_(model, state_dict) -> None:
    """Fold a trainer-passed (possibly Zero3-sharded) state_dict into student.

    ms-swift TrainerMixin.save_pretrained -> tuners_map.save_pretrained with
    state_dict=trainable-only. Fold it back so merged weights are current.
    """
    if not state_dict:
        return
    try:
        student = getattr(model, "student", model)
        own = dict(student.named_parameters())
        for k, v in state_dict.items():
            short = k.split("student.")[-1] if "student." in k else k
            key = short if short in own else (k if k in own else None)
            if key is not None:
                try:
                    if tuple(own[key].shape) == tuple(v.shape):
                        with torch.no_grad():
                            own[key].copy_(v.to(own[key].device, own[key].dtype))
                except Exception:
                    continue
    except Exception:
        pass


def _resolve_teacher(student: nn.Module, teacher_id: str) -> nn.Module:
    """Reuse weights when teacher == student to avoid double VRAM.

    - Same local dir / same HF id as the student model: deep-copy the already
      loaded student (ms-swift loaded it with device_map/quantization right),
      so no second from_pretrained (no OOM, no dtype/device mismatch).
    - Otherwise load the open-source teacher on CPU (bf16) for SVD basis +
      KL logits; moved to student device on demand in get_teacher_outputs.
    """
    try:
        student_id = getattr(student, "model_dir", None) or getattr(student, "name_or_path", None)
    except Exception:
        student_id = None
    try:
        if student_id and teacher_id and os.path.realpath(str(student_id)) == os.path.realpath(str(teacher_id)):
            t = deepcopy(student)
            for p in t.parameters():
                p.requires_grad_(False)
            return t.eval()
    except Exception:
        pass
    if student_id and teacher_id and str(student_id) == str(teacher_id):
        t = deepcopy(student)
        for p in t.parameters():
            p.requires_grad_(False)
        return t.eval()
    try:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_id, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="cpu", low_cpu_mem_usage=True)
    except Exception:
        teacher = deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval()


def _hidden_of(mod: nn.Module) -> int:
    cfg = getattr(mod, "config", None)
    tc = getattr(cfg, "text_config", None) or cfg
    try:
        return int(getattr(tc, "hidden_size", 0) or 0)
    except Exception:
        return 0


def _shrink_student_(student: nn.Module, teacher: nn.Module, r: int) -> nn.Module:
    """Replace a full-width student with a width-r LRC student (slice init).

    TWYD trains the deployed width-reduced model: hidden r, d_ff inherited from
    the teacher, heads = r / head_dim (LowRankClone target_hidden_size style;
    scripts/train_dense_standalone.py:build_student_from_teacher is the
    reference). Warm start = slice the teacher weights (warm_start='slice').
    Swift-set attrs (model_meta/model_dir/...) are carried onto the new model
    so the stock swift pt/sft loop keeps working.
    """
    old_cfg = getattr(student, "config", None)
    tc = getattr(old_cfg, "text_config", None) or old_cfg
    H = _hidden_of(student)
    heads = int(getattr(tc, "num_attention_heads", 8) or 8)
    kv = int(getattr(tc, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(tc, "head_dim", None) or ((H // heads) if heads else 64))
    if head_dim <= 0 or r % head_dim != 0:
        raise ValueError(
            f"[twyd] student_hidden r={r} not divisible by head_dim={head_dim}; "
            "pick TWYD_R that is a multiple of the teacher head_dim")
    new_heads = r // head_dim
    # KV heads must divide query heads for GQA (SDPA requirement). Pick the
    # largest divisor of new_heads that does not exceed the teacher's kv heads.
    _kv_cap = min(kv, new_heads)
    new_kv = next((d for d in range(_kv_cap, 0, -1) if new_heads % d == 0), 1)
    cfg = deepcopy(old_cfg)
    target = getattr(cfg, "text_config", None)
    target = target if target is not None else cfg
    target.hidden_size = r
    target.num_attention_heads = new_heads
    if getattr(target, "num_key_value_heads", None):
        target.num_key_value_heads = new_kv
    from transformers import AutoModelForCausalLM
    try:
        new = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    except TypeError:
        new = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        tsd = dict(teacher.named_parameters())
        for n, p in new.named_parameters():
            t = tsd.get(n)
            if t is None:
                continue
            if t.dim() != p.dim() or any(ps > tsz for ps, tsz in zip(p.shape, t.shape)):
                continue
            idx = tuple(slice(0, s) for s in p.shape)
            try:
                p.copy_(t[idx].to(p.dtype))
            except Exception:
                continue
    # carry swift attrs from the old student (plain attrs, not params/modules)
    for attr in ("model_meta", "model_dir", "model_info", "model_type",
                 "name_or_path"):
        if attr in student.__dict__:
            setattr(new, attr, student.__dict__[attr])
    for attr in ("generation_config", "origin_generation_config"):
        if attr in student.__dict__:
            try:
                setattr(new, attr, deepcopy(student.__dict__[attr]))
            except Exception:
                pass
    try:
        ref = next(student.parameters())
        new = new.to(device=ref.device, dtype=ref.dtype)
    except StopIteration:
        pass
    print(f"[twyd] width shrink: student hidden {H} -> {r} "
          f"(heads {heads}->{new_heads}, kv {kv}->{new_kv}, d_ff inherited)")
    return new


def _twyd_kw(args) -> dict:
    """Read twyd hyper-params.

    Priority: TWYD_* env vars > legacy args.tuner_kwargs dict (old swift).
    New swift (4.x) has no tuner_kwargs field, so env vars are canonical:
      TWYD_TEACHER, TWYD_R, TWYD_STAGE, TWYD_WARM_START, TWYD_PT_CKPT,
      TWYD_T, TWYD_WKL, TWYD_WNTP, TWYD_LAM, TWYD_USE_AUX,
      TWYD_TARGET_MODULES (comma-separated), TWYD_OUTLIER_KEEP,
      TWYD_RIDGE_LAMBDA, TWYD_MAX_AUX_LAYERS.
    """
    import os as _os

    kw = dict(getattr(args, "tuner_kwargs", None) or {})
    env_map = {
        "teacher_model": _os.environ.get("TWYD_TEACHER", None),
        "student_hidden": _os.environ.get("TWYD_R", None),
        "stage": _os.environ.get("TWYD_STAGE", None),
        "warm_start": _os.environ.get("TWYD_WARM_START", None),
        "pt_ckpt": _os.environ.get("TWYD_PT_CKPT", None),
        "kl_temperature": _os.environ.get("TWYD_T", None),
        "kl_weight": _os.environ.get("TWYD_WKL", None),
        "ntp_weight": _os.environ.get("TWYD_WNTP", None),
        "aux_weight": _os.environ.get("TWYD_LAM", None),
        "use_aux": _os.environ.get("TWYD_USE_AUX", None),
        "target_modules": _os.environ.get("TWYD_TARGET_MODULES", None),
        "outlier_keep": _os.environ.get("TWYD_OUTLIER_KEEP", None),
        "ridge_lambda": _os.environ.get("TWYD_RIDGE_LAMBDA", None),
        "lora_rank": _os.environ.get("TWYD_LORA_RANK", None),
        "lora_alpha": _os.environ.get("TWYD_LORA_ALPHA", None),
    }
    for k, v in env_map.items():
        if v is None or (isinstance(v, str) and v == ""):
            continue
        kw[k] = v
    # Type coercion for env-sourced strings.
    def _num(x, f=float):
        try:
            return f(x)
        except Exception:
            return x

    for k in ("student_hidden", "outlier_keep", "lora_rank"):
        if k in kw and isinstance(kw[k], str):
            try:
                kw[k] = int(kw[k])
            except Exception:
                pass
    for k in ("lora_alpha",):
        if k in kw and isinstance(kw[k], str):
            try:
                kw[k] = float(kw[k])
            except Exception:
                pass
    for k in ("kl_temperature", "kl_weight", "ntp_weight", "aux_weight", "ridge_lambda"):
        if k in kw and isinstance(kw[k], str):
            kw[k] = _num(kw[k])
    if "use_aux" in kw and isinstance(kw["use_aux"], str):
        kw["use_aux"] = kw["use_aux"].strip().lower() not in ("0", "false", "no", "off")
    if "target_modules" in kw and isinstance(kw["target_modules"], str):
        kw["target_modules"] = [s.strip() for s in kw["target_modules"].split(",") if s.strip()]
    mx = _os.environ.get("TWYD_MAX_AUX_LAYERS", None)
    if mx is not None and mx != "":
        try:
            kw["max_aux_layers"] = int(mx)
        except Exception:
            pass
    return kw


def _prepare_model_impl(args, model, branch: str):
    from .config import TwydLRCConfig as Cfg
    kw = _twyd_kw(args)
    cfg = Cfg(branch=branch,
              teacher_model=kw.get("teacher_model"),
              student_hidden=kw.get("student_hidden"),
              stage=str(kw.get("stage", "pt")).lower(),
              warm_start=str(kw.get("warm_start", "slice")).lower(),
              target_modules=kw.get("target_modules", ["gate_proj", "up_proj", "down_proj"]),
              kl_temperature=kw.get("kl_temperature", 40.0),
              kl_weight=kw.get("kl_weight", 1.0),
              ntp_weight=kw.get("ntp_weight", 1.0),
              aux_weight=kw.get("aux_weight", 0.2),
              use_aux=kw.get("use_aux", True),
              aux_loss_type=kw.get("aux_loss_type", "mseloss"),
              ridge_lambda=kw.get("ridge_lambda"),
              outlier_keep=kw.get("outlier_keep", 0),
              lora_rank=kw.get("lora_rank", 0),
              lora_alpha=kw.get("lora_alpha", 16.0))
    cfg.validate()
    teacher_id = cfg.teacher_model or getattr(args, "model", None)
    if teacher_id is None:
        raise ValueError("twyd needs tuner_kwargs.teacher_model")
    student = model
    teacher = _resolve_teacher(student, teacher_id)
    teacher.eval()
    # Width shrink: TWYD trains the deployed width-r model. If the --model
    # student is still full-width, rebuild it at hidden=r (slice warm start
    # from the teacher) BEFORE the PT-ckpt warm start and factorized rebuild,
    # so PT ckpt (width-r) tensors match and gate/up are (d_ff, r).
    if not cfg.student_hidden:
        cfg.student_hidden = max(64, _hidden_of(teacher) // 2)
    r = int(cfg.student_hidden)
    if _hidden_of(student) != r:
        student = _shrink_student_(student, teacher, r)
    # SFT warm start: load merged PT ckpt into the (possibly shrunk) student
    # BEFORE factorized rebuild, so Z_col/W0 inherit PT-learned weights
    # instead of the raw slice init. Teacher for SFT stays the open teacher.
    pt_ckpt = kw.get("pt_ckpt", None)
    if cfg.stage == "sft" and cfg.warm_start == "pt_ckpt" and pt_ckpt:
        try:
            n = _load_pt_merged_(student, pt_ckpt)
            print(f"[twyd] SFT warm start: loaded {n} tensors from PT ckpt {pt_ckpt}")
        except Exception as e:
            print(f"[twyd] SFT warm start failed ({e}); falling back to slice init")
    meta = _build_factorized_student(student, teacher, cfg, branch=branch)
    wrapper = _TwydWrapper(student, teacher, cfg)
    wrapper.twyd_meta = meta
    _freeze_for_branch(wrapper, cfg)
    return wrapper


class DenseLRCAdapter(_TunerBase):
    branch = "dense"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "dense")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        _merge_state_dict_(model, state_dict)
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        # Resume path: model_id is a twyd training ckpt dir (adapter weights
        # saved by merge_and_save + trainer state). Reload merged weights into
        # the live student so --resume_from_checkpoint keeps Z_col/Z_perp values.
        # Deploy path: model_id is a merged HF dir -> load directly.
        is_trainable = kwargs.get("is_trainable", False)
        try:
            import os as _os
            if _os.path.isdir(str(model_id)):
                files = set(os.listdir(str(model_id)))
                student = getattr(model, "student", model)
                if "model.safetensors" in files or "pytorch_model.bin" in files:
                    try:
                        n = _load_pt_merged_(student, str(model_id))
                    except Exception:
                        n = 0
                    if n > 0 and (is_trainable or kwargs.get("adapter_name", None) is None):
                        return model
            from transformers import AutoModelForCausalLM
            import torch as _t
            loaded = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=getattr(model, "dtype", _t.bfloat16), trust_remote_code=True)
            return loaded
        except Exception:
            return model


class CoreLRCAdapter(_TunerBase):
    branch = "core"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "core")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        _merge_state_dict_(model, state_dict)
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        return DenseLRCAdapter.from_pretrained(model, model_id, **kwargs)


class PlainLRCAdapter(_TunerBase):
    branch = "plain"

    @staticmethod
    def prepare_model(args, model):
        return _prepare_model_impl(args, model, "plain")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        from .merge import merge_and_save
        _merge_state_dict_(model, state_dict)
        merge_and_save(model, save_directory, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        return model

