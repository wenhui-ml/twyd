"""TWYD LRC distillation config."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TwydLRCConfig:
    """Config shared by tuner / loss / init / merge.

    branch: "dense" | "core" | "plain" (plain = matched LRC baseline, col path only).
    teacher_model: HF id or local path of the teacher (frozen, for KL + SVD basis).
    student_hidden: r = d_model^(S), e.g. 1536 for Llama3.2-3B->1.5B.
    target_modules: MLP projections to open. Default gate/up/down only (paper App. B).
    kl_temperature: T=40 (paper A.5). kl_weight=wKL=1.0, ntp_weight=wNTP=1.0.
    aux_weight: lambda_aux=0.2. use_aux toggles hidden/attn/mlp alignment.
    aux_loss_type: "mseloss" | "l1".
    ridge_lambda: None -> auto tr(C^TC)/r (Eq.4); else explicit.
    outlier_keep: k outlier passthrough rows for Qwen P (Eq.5, default 0 = plain SVD).
    """

    branch: str = "core"
    teacher_model: Optional[str] = None
    student_hidden: Optional[int] = None
    stage: str = "pt"
    warm_start: str = "slice"   # slice | pt_ckpt | ridge | ridge_ckpt | base
    target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    kl_temperature: float = 40.0
    kl_weight: float = 1.0
    ntp_weight: float = 1.0
    aux_weight: float = 0.2
    use_aux: bool = True
    aux_loss_type: str = "mseloss"  # mseloss | l1 | logit_lens
    aux_layers: str = ""            # explicit mid-layer indices for L_align, e.g. "6,12,16,19"
    kl_topk: int = 0                # >0 enables Top-K + tail-bucket KL
    rho_keep: float = 1.0           # <1.0 enables Rho-1 excess-loss token selection
    rho_min_weight: float = 0.1     # floor weight for non-selected tokens in Rho-1
    unfreeze_attn: bool = False     # unfreeze q/k/v/o_proj in active layers (Full-Bus)
    unfreeze_norm: bool = False     # unfreeze RMSNorm in active layers
    writer_lr_scale: float = 1.0    # LR multiplier for Writer projections (down_proj, o_proj)
    writer_grad_scale: float = 1.0  # grad multiplier for Writer projections before global clip
    writer_max_norm: float = 0.0    # per-tensor grad norm clamp for Writer projections
    ridge_lambda: Optional[float] = None
    ridge_perp: str = "residual"  # zero | residual | lowrank (ridge warm start)
    master_p: str = "none"        # none | embed_svd (project before ridge)
    calib_ids: Optional[str] = None
    calib_num_seqs: int = 128
    calib_seq_len: int = 512
    calib_batch_size: int = 2
    calib_joint: bool = True
    outlier_keep: int = 0
    lora_rank: int = 0          # 0 = full dense-lrc (no LoRA); >0 = LoRA-Dense-LRC
    lora_alpha: float = 16.0    # LoRA scaling = alpha / rank
    lora_mode: str = "dense"    # "dense" | "core": LoRA 挂在 deployed 权重还是补空间
    lora_layers: str = ""       # e.g. "0-8": 只在这些层挂 LoRA，其余保持 plain
    lora_modules: str = ""      # e.g. "gate_proj,up_proj": 只训这些投影（空=全部）
    lora_lr: Optional[float] = None  # 补空间 LoRA 独立学习率（None=跟随全局 lr）
    freeze_col: bool = False    # CORE: 冻结 plain-LRC col path，只训补空间（CORE-LR）
    rank_map: str = ""          # JSON dict or path: module name -> per-module k
    factorized: bool = False    # CORE-LR: cached W_col + low-rank delta forward
    perp_basis: str = "teacher"  # teacher | col (col = P6 equal-parameter control)

    def validate(self) -> None:
        assert self.branch in ("dense", "core", "plain"), f"branch={self.branch}"
        assert self.aux_loss_type in ("mseloss", "l1", "logit_lens"), self.aux_loss_type
        assert self.stage in ("pt", "sft"), f"stage={self.stage}"
        assert self.warm_start in ("slice", "pt_ckpt", "ridge", "ridge_ckpt", "base"), \
            f"warm_start={self.warm_start}"
        assert self.ridge_perp in ("zero", "residual", "lowrank"), \
            f"ridge_perp={self.ridge_perp}"
        assert self.master_p in ("none", "embed_svd"), f"master_p={self.master_p}"
        assert self.lora_mode in ("dense", "core"), f"lora_mode={self.lora_mode}"
        assert self.perp_basis in ("teacher", "col"), f"perp_basis={self.perp_basis}"


def parse_layer_range(spec: str, n_layers: int) -> set:
    """Parse "0-8" / "0-3,10-15" into a set of layer indices."""
    out: set = set()
    if not spec or not str(spec).strip():
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
            except Exception:
                continue
            for i in range(min(lo, hi), max(lo, hi) + 1):
                if 0 <= i < n_layers:
                    out.add(i)
        else:
            try:
                i = int(part)
            except Exception:
                continue
            if 0 <= i < n_layers:
                out.add(i)
    return out


def layer_idx_of(mod_name: str) -> Optional[int]:
    """Extract layer index from "model.layers.7.gate_proj" style names."""
    import re as _re
    m = _re.search(r"layers\.(\d+)", mod_name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_rank_map(spec, default: int = 0) -> dict:
    """Parse ``TWYD_RANK_MAP`` into ``{module_key: k}``.

    Accepted forms:
      - JSON object string: ``'{"layers.0.mlp.gate_proj": 128, "down_proj": 64}'``
      - path to a JSON file with the same object
    Keys may be full module names, ``layers.N.mlp.proj`` suffixes, or a bare
    projection name (``gate_proj``/``up_proj``/``down_proj``).  Resolution is
    exact first, then longest suffix, then short name.
    """
    import json
    import os

    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        raw = spec
    else:
        text = str(spec).strip()
        if os.path.isfile(text):
            with open(text, "r") as f:
                raw = json.load(f)
        else:
            raw = json.loads(text)
    out = {}
    for k, v in (raw or {}).items():
        try:
            iv = int(v)
        except Exception:
            continue
        if iv >= 0:
            out[str(k)] = iv
    return out


def module_rank(rank_map: dict, mod_name: str, default: int = 0) -> int:
    """Resolve a per-module rank with exact/suffix/layer+projection matching."""
    if not rank_map:
        return int(default)
    name = str(mod_name)
    if name in rank_map:
        return int(rank_map[name])
    short = name.split(".")[-1]
    # longest suffix match: "layers.0.mlp.gate_proj" matches
    # "model.layers.0.mlp.gate_proj"
    for key in sorted(rank_map, key=len, reverse=True):
        if name.endswith("." + key):
            return int(rank_map[key])
    # "layers.0.gate_proj" must still match "model.layers.0.mlp.gate_proj"
    import re as _re
    for key in sorted(rank_map, key=len, reverse=True):
        m = _re.search(r"layers\.(\d+)\.", str(key))
        key_short = str(key).split(".")[-1]
        if (m and key_short == short and name.endswith("." + short)
                and f"layers.{m.group(1)}." in name):
            return int(rank_map[key])
    if short in rank_map:
        return int(rank_map[short])
    return int(default)
