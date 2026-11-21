"""ms-swift loss_map plugin: lrc_distill (PT + SFT stages).

PT (stage="pt"): labels == input_ids, every position distills.
SFT (stage="sft"): prompt/pad masked to -100, KL+CE+aux all respect mask.
"""
import os
import time

import torch
import torch.nn.functional as F


_PROF = {"n": 0, "t_fwd": 0.0, "t_teacher": 0.0, "t_kl": 0.0, "t_aux": 0.0,
         "t_ce": 0.0}
# The composite loss is CE + T^2*KL with T=40, so the KL term (~178) swamps CE (~2.5):
# a 1% CE improvement moves the number by 0.03 and looks like "no learning". Print the
# two components every `TWYD_LOG_EVERY` micro-batches (default 320 = 10 steps at gas=32)
# so progress is visible at all.
_CALLS = {"n": 0}


def fit_hidden_projection(A: torch.Tensor, B: torch.Tensor, max_rows: int = 1024):
    """Least-squares P mapping A(N, r) -> B(N, H), for N >= r *and* N < r.

    ``torch.linalg.lstsq`` only accepts overdetermined systems on CUDA/ROCm
    ("only overdetermined systems ... are allowed on CUDA").  With seq<=1024 the
    number of supervised positions is usually far below the student width r, so
    the old call raised, the outer ``except`` swallowed it and the auxiliary
    alignment term was silently dropped.  Underdetermined systems now get the
    minimum-norm pseudo-inverse solution.
    """
    n = int(A.shape[0])
    if n == 0:
        return None
    if n > max_rows:
        step = max(1, n // max_rows)
        A, B = A[::step][:max_rows], B[::step][:max_rows]
    A = A.detach().float()
    B = B.detach().float()
    try:
        if A.shape[0] >= A.shape[1]:
            return torch.linalg.lstsq(A, B).solution
        return torch.linalg.pinv(A) @ B
    except Exception:
        return None


def _prof_on() -> bool:
    return os.environ.get("TWYD_PROFILE", "") not in ("", "0", "false", "False")


def _tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _tock(t0):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


try:
    from swift.loss.base import BaseLoss as _BaseLoss
except Exception:  # standalone test without ms-swift installed
    _BaseLoss = object


class LrcDistillLoss(_BaseLoss):
    def __init__(self, args, trainer):
        try:
            super().__init__(args, trainer)
        except Exception:
            self.args = args
            self.trainer = trainer
        try:
            from .tuner import _twyd_kw
            kw = dict(_twyd_kw(args))
        except Exception:
            kw = dict(getattr(args, "tuner_kwargs", None) or {})
        self.kw = kw
        self.T = float(kw.get("kl_temperature", 40.0))
        self.wKL = float(kw.get("kl_weight", 1.0))
        self.wNTP = float(kw.get("ntp_weight", 1.0))
        self.lam = float(kw.get("aux_weight", 0.2))
        self.use_aux = bool(kw.get("use_aux", True))
        self._aux_norm = bool(kw.get("aux_norm", False))
        self.aux_loss_type = str(kw.get("aux_loss_type", "mseloss") or "mseloss").lower()
        self.aux_layers = str(kw.get("aux_layers", "") or "")
        self.kl_topk = int(kw.get("kl_topk", 0) or 0)
        self.rho_keep = float(kw.get("rho_keep", 1.0) if kw.get("rho_keep") is not None else 1.0)
        self.rho_min_weight = float(kw.get("rho_min_weight", 0.1) if kw.get("rho_min_weight") is not None else 0.1)
        self.stage = str(kw.get("stage", "pt")).lower()
        self._wire_optimizer_groups(trainer)

    def _wire_optimizer_groups(self, trainer):
        """Hook trainer.optimizer_callback so Writer/Reader & LoRA LR groups take effect."""
        try:
            cb = getattr(trainer, "optimizer_callback", None)
            if cb is None or getattr(cb, "_twyd_opt_wired", False):
                return
            orig_create = cb.create_optimizer
            loss_self = self

            def _create_optimizer(model=None):
                m_target = model if model is not None else getattr(trainer, "model", None)
                w = loss_self._unwrap(m_target) if m_target is not None else None
                fn = getattr(w, "create_optimizer_param_groups", None)
                if callable(fn):
                    t_args = getattr(trainer, "args", None) or getattr(cb, "args", None)
                    lr = float(getattr(t_args, "learning_rate", 1e-4) or 1e-4)
                    wd = float(getattr(t_args, "weight_decay", 0.0) or 0.0)
                    groups = fn(lr=lr, weight_decay=wd)
                    if groups:
                        from transformers.trainer import Trainer as HfTrainer
                        opt_cls, opt_kw = HfTrainer.get_optimizer_cls_and_kwargs(t_args)
                        print(f"[twyd] custom optimizer param groups: {len(groups)} groups "
                              f"(lrs={[g.get('lr') for g in groups]})", flush=True)
                        return opt_cls(groups, **opt_kw)
                return orig_create(model=model)

            cb.create_optimizer = _create_optimizer
            cb._twyd_opt_wired = True
        except Exception:
            pass

    def _unwrap(self, model):
        # Unwrap Swift/Peft chains, but STOP at _TwydWrapper (the layer that
        # carries twyd_cfg / get_teacher_outputs / _twyd_last_inputs).
        # Our wrapper exposes .model -> student for ms-swift compat, so a
        # naive .model chase would overshoot past the wrapper.
        m = model
        for _ in range(5):
            try:
                has_cfg = getattr(m, "twyd_cfg", None) is not None
            except Exception:
                has_cfg = False
            if has_cfg:
                return m
            inner = getattr(m, "model", None)
            if inner is None or inner is m:
                break
            m = inner() if callable(inner) else inner
        return m

    @staticmethod
    def _chunk_topk_kl_ce(
        s_c: torch.Tensor,
        top_idx_c: torch.Tensor,
        logp_t_top_c: torch.Tensor,
        p_t_top_c: torch.Tensor,
        p_t_tail_c: torch.Tensor,
        y_c: torch.Tensor | None,
        T: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused Top-K + Tail-Bucket KL and CE on one sequence chunk (recomputed under checkpoint)."""
        V_full = int(s_c.size(-1))
        logp_s = F.log_softmax(s_c.float() / T, dim=-1)
        logp_s_top = torch.gather(logp_s, dim=-1, index=top_idx_c)
        p_s_top = logp_s_top.exp()
        p_s_tail = (1.0 - p_s_top.sum(dim=-1)).clamp_min(1e-7)
        kl_top = (p_t_top_c * (logp_t_top_c - logp_s_top)).sum(dim=-1)
        kl_tail = p_t_tail_c * (p_t_tail_c.log() - p_s_tail.log())
        kl_c = (kl_top + kl_tail).clamp_min(0.0) * (T * T)
        if y_c is None:
            return kl_c, None
        if abs(T - 1.0) < 1e-6:
            safe_y = y_c.clamp_min(0)
            ce_c = -torch.gather(logp_s, dim=-1, index=safe_y.unsqueeze(-1)).squeeze(-1)
            ce_c = torch.where(y_c != -100, ce_c, torch.zeros_like(ce_c))
        else:
            ce_c = F.cross_entropy(
                s_c.float().reshape(-1, V_full), y_c.reshape(-1),
                reduction="none", ignore_index=-100,
            ).view_as(y_c)
        return kl_c, ce_c

    def _fused_topk_distill(
        self,
        shift_s: torch.Tensor,
        shift_t: torch.Tensor,
        shift_y: torch.Tensor | None,
        T: float,
        topk: int,
        use_rho: bool,
        rho_keep: float,
        rho_min_w: float,
        chunk: int = 256,
    ):
        """Memory-optimal fused Top-K KL + CE + Rho-1 over 128k vocab.

        Compresses the [B, S, V] teacher logits into [B, S, K] targets in no_grad
        (freeing ~1 GiB immediately) and runs each 256-token student chunk under
        non-reentrant activation checkpointing so peak loss VRAM is ~0.25 GiB
        instead of ~4.2 GiB while remaining mathematically equivalent.
        """
        from torch.utils.checkpoint import checkpoint as _ckpt

        V = min(int(shift_s.size(-1)), int(shift_t.size(-1)))
        S = int(shift_s.size(1))
        B = int(shift_s.size(0))
        s_v = shift_s[..., :V]
        t_v = shift_t[..., :V]

        top_idxs, logp_t_tops, p_t_tops, p_t_tails = [], [], [], []
        ce_t_chunks, ce_s_det_chunks, kl_det_chunks = [], [], []

        with torch.no_grad():
            for st in range(0, S, chunk):
                ed = min(S, st + chunk)
                t_c = t_v[:, st:ed, :]
                logp_t = F.log_softmax(t_c.float() / T, dim=-1)
                logp_t_top, top_idx = torch.topk(logp_t, k=topk, dim=-1)
                p_t_top = logp_t_top.exp()
                p_t_tail = (1.0 - p_t_top.sum(dim=-1)).clamp_min(1e-7)
                top_idxs.append(top_idx)
                logp_t_tops.append(logp_t_top)
                p_t_tops.append(p_t_top)
                p_t_tails.append(p_t_tail)

                if use_rho and shift_y is not None:
                    y_c = shift_y[:, st:ed]
                    s_c = s_v[:, st:ed, :].detach()
                    if abs(T - 1.0) < 1e-6:
                        safe_y = y_c.clamp_min(0)
                        ce_t_c = -torch.gather(logp_t, dim=-1, index=safe_y.unsqueeze(-1)).squeeze(-1)
                        ce_t_c = torch.where(y_c != -100, ce_t_c, torch.zeros_like(ce_t_c))
                    else:
                        ce_t_c = F.cross_entropy(
                            t_c.float().reshape(-1, V), y_c.reshape(-1),
                            reduction="none", ignore_index=-100,
                        ).view_as(y_c)
                    kl_det_c, ce_s_det_c = self._chunk_topk_kl_ce(
                        s_c, top_idx, logp_t_top, p_t_top, p_t_tail, y_c, T
                    )
                    ce_t_chunks.append(ce_t_c)
                    ce_s_det_chunks.append(ce_s_det_c)
                    kl_det_chunks.append(kl_det_c)

        rho_w = None
        if use_rho and shift_y is not None:
            valid = (shift_y != -100)
            if S <= 1:
                rho_w = valid.float()
            else:
                with torch.no_grad():
                    ce_s_det = torch.cat(ce_s_det_chunks, dim=1)
                    ce_t_det = torch.cat(ce_t_chunks, dim=1)
                    kl_nats = torch.cat(kl_det_chunks, dim=1) / max(T * T, 1.0)
                    score = (ce_s_det - ce_t_det) + kl_nats
                    # Exclude padded positions (labels == -100) before quantile / topk selection.
                    flat_t = ce_t_det[valid]
                    if flat_t.numel() >= 16:
                        q90 = torch.quantile(flat_t, 0.90)
                        score = torch.where(ce_t_det > q90, torch.full_like(score, -1e9), score)
                    score = torch.where(valid, score, torch.full_like(score, -1e9))
                    rho_w = torch.where(valid, torch.full_like(score, float(rho_min_w)), torch.zeros_like(score))
                    for b in range(B):
                        n_val = int(valid[b].sum().item())
                        if n_val <= 0:
                            continue
                        k_keep = max(1, min(n_val, int(round(n_val * float(rho_keep)))))
                        _, top_pos = torch.topk(score[b], k=k_keep, dim=-1)
                        rho_w[b].scatter_(-1, top_pos, 1.0)
                    rho_w = rho_w * valid.float()

        # Drop references to teacher chunks before running grad-tracked student chunks.
        del t_v, ce_t_chunks, ce_s_det_chunks, kl_det_chunks

        kl_outs, ce_outs = [], []
        use_ckpt = bool(s_v.requires_grad and torch.is_grad_enabled())
        need_ce = shift_y is not None
        for idx_c, st in enumerate(range(0, S, chunk)):
            ed = min(S, st + chunk)
            s_c = s_v[:, st:ed, :]
            y_c = shift_y[:, st:ed] if need_ce else None
            ti_c = top_idxs[idx_c]
            lpt_c = logp_t_tops[idx_c]
            pt_c = p_t_tops[idx_c]
            ptail_c = p_t_tails[idx_c]
            if use_ckpt:
                if need_ce:
                    kl_c, ce_c = _ckpt(
                        lambda _s, _ti, _lpt, _pt, _ptail, _y: self._chunk_topk_kl_ce(
                            _s, _ti, _lpt, _pt, _ptail, _y, T
                        ),
                        s_c, ti_c, lpt_c, pt_c, ptail_c, y_c,
                        use_reentrant=False,
                    )
                    ce_outs.append(ce_c)
                else:
                    kl_c = _ckpt(
                        lambda _s, _ti, _lpt, _pt, _ptail: self._chunk_topk_kl_ce(
                            _s, _ti, _lpt, _pt, _ptail, None, T
                        )[0],
                        s_c, ti_c, lpt_c, pt_c, ptail_c,
                        use_reentrant=False,
                    )
            else:
                kl_c, ce_c = self._chunk_topk_kl_ce(s_c, ti_c, lpt_c, pt_c, ptail_c, y_c, T)
                if need_ce:
                    ce_outs.append(ce_c)
            kl_outs.append(kl_c)

        per_pos_kl = kl_outs[0] if len(kl_outs) == 1 else torch.cat(kl_outs, dim=1)
        ce_per_pos = (ce_outs[0] if len(ce_outs) == 1 else torch.cat(ce_outs, dim=1)) if need_ce else None
        return per_pos_kl, ce_per_pos, rho_w

    def _per_pos_kl(self, shift_s, shift_t, T, chunk=256, topk: int = 0):
        """Per-position KL(t||s) in fp32, chunked over the sequence.

        When ``topk > 0``, computes exact Top-K + 1 Tail-Bucket KL:
            KL_topK(p_T || p_S) = sum_{i in TopK} p_T(i) log(p_T(i)/p_S(i))
                                  + p_T(tail) log(p_T(tail)/p_S(tail))
        collapsing the (V - K) low-probability tail tokens into one bucket so
        gradients focus on separating competing top-K concepts rather than
        smoothing 128k background noise tokens.
        """
        V = min(shift_s.size(-1), shift_t.size(-1))
        use_topk = int(topk or 0)
        outs = []
        for s_c, t_c in zip(shift_s[..., :V].split(chunk, dim=1),
                            shift_t[..., :V].split(chunk, dim=1)):
            logp_s = F.log_softmax(s_c.float() / T, dim=-1)
            logp_t = F.log_softmax(t_c.float() / T, dim=-1)
            if 0 < use_topk < V:
                logp_t_top, top_idx = torch.topk(logp_t, k=use_topk, dim=-1)
                logp_s_top = torch.gather(logp_s, dim=-1, index=top_idx)
                p_t_top = logp_t_top.exp()
                p_s_top = logp_s_top.exp()
                p_t_tail = (1.0 - p_t_top.sum(dim=-1)).clamp_min(1e-7)
                p_s_tail = (1.0 - p_s_top.sum(dim=-1)).clamp_min(1e-7)
                kl_top = (p_t_top * (logp_t_top - logp_s_top)).sum(dim=-1)
                kl_tail = p_t_tail * (p_t_tail.log() - p_s_tail.log())
                kl_pos = (kl_top + kl_tail).clamp_min(0.0) * (T * T)
                outs.append(kl_pos)
            else:
                outs.append((logp_t.exp() * (logp_t - logp_s)).sum(dim=-1) * (T * T))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    def _masked_mean(self, per_pos, shift_y, loss_scale=None):
        mask = (shift_y != -100).to(per_pos.dtype)
        w = mask
        if torch.is_tensor(loss_scale):
            ls = loss_scale
            try:
                if ls.dim() == 2 and ls.shape[1] == shift_y.shape[1] + 1:
                    ls = ls[..., 1:].contiguous()
                w = mask * ls.to(mask.dtype).reshape(mask.shape)
            except Exception:
                w = mask
        return (per_pos * w).sum() / w.sum().clamp_min(1.0)

    def _rho_token_weights(self, shift_s, shift_t, shift_y, per_pos_kl, T: float,
                           rho_keep: float, rho_min_weight: float):
        """Compute per-token weights via Rho-1 / Teacher-Student Manifold Divergence.

        Tokens where the teacher is confident (not random web noise) AND the
        student lags behind the teacher (high excess CE + KL divergence) receive
        weight 1.0; trivial / already-mastered tokens receive ``rho_min_weight``.
        Zero extra forward passes required.
        """
        B, S, V = shift_s.shape
        if S <= 1:
            return torch.ones_like(per_pos_kl)
        with torch.no_grad():
            ce_s_chunks, ce_t_chunks = [], []
            V_t = int(shift_t.size(-1))
            for st in range(0, S, 256):
                ed = min(S, st + 256)
                yc = shift_y[:, st:ed].reshape(-1)
                ce_s_chunks.append(F.cross_entropy(
                    shift_s[:, st:ed, :].detach().float().reshape(-1, V), yc, reduction="none"
                ).view(B, ed - st))
                ce_t_chunks.append(F.cross_entropy(
                    shift_t[:, st:ed, :].detach().float().reshape(-1, V_t), yc, reduction="none"
                ).view(B, ed - st))
            ce_s_det = torch.cat(ce_s_chunks, dim=1)
            ce_t_det = torch.cat(ce_t_chunks, dim=1)
            kl_nats = per_pos_kl.detach().float() / max(T * T, 1.0)
            score = (ce_s_det - ce_t_det) + kl_nats
            # Filter out high-entropy teacher noise tokens (top 10% highest teacher CE)
            flat_t = ce_t_det.reshape(-1)
            if flat_t.numel() >= 16:
                q90 = torch.quantile(flat_t, 0.90)
                score = torch.where(ce_t_det > q90, torch.full_like(score, -1e9), score)
            w = torch.full_like(score, float(rho_min_weight))
            k_keep = max(1, min(S, int(round(S * float(rho_keep)))))
            _, top_pos = torch.topk(score, k=k_keep, dim=-1)
            w.scatter_(-1, top_pos, 1.0)
        return w.to(per_pos_kl.dtype)

    @staticmethod
    def _project_logit_lens(mod, h: torch.Tensor):
        """Project hidden state h [B, S, d] to vocab logits via detached norm + lm_head."""
        inner = getattr(mod, "model", mod)
        norm = getattr(inner, "norm", None)
        lm_head = getattr(mod, "lm_head", None)
        if lm_head is None and hasattr(mod, "get_output_embeddings"):
            try:
                lm_head = mod.get_output_embeddings()
            except Exception:
                lm_head = None
        if lm_head is None or not hasattr(lm_head, "weight"):
            return None
        x = h
        if norm is not None and hasattr(norm, "weight"):
            eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)) or 1e-5)
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            nw = norm.weight.detach().to(device=x.device, dtype=x.dtype)
            x = xf.to(x.dtype) * nw
        hw = lm_head.weight.detach().to(device=x.device, dtype=x.dtype)
        return F.linear(x, hw)

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        raw_model = self.trainer.model
        _p = _prof_on()
        _t0 = _tick() if _p else 0.0
        m = self._unwrap(raw_model)
        twcfg = getattr(m, "twyd_cfg", None) or getattr(raw_model, "twyd_cfg", None)
        T, wKL, wNTP, lam = self.T, self.wKL, self.wNTP, self.lam
        kl_topk = self.kl_topk
        rho_keep = self.rho_keep
        rho_min_w = self.rho_min_weight
        aux_type = self.aux_loss_type
        aux_layers_spec = self.aux_layers
        if twcfg is not None:
            T, wKL, wNTP, lam = twcfg.kl_temperature, twcfg.kl_weight, twcfg.ntp_weight, twcfg.aux_weight
            use_aux = twcfg.use_aux
            kl_topk = int(getattr(twcfg, "kl_topk", kl_topk) or 0)
            rho_keep = float(getattr(twcfg, "rho_keep", rho_keep) if getattr(twcfg, "rho_keep", None) is not None else rho_keep)
            rho_min_w = float(getattr(twcfg, "rho_min_weight", rho_min_w) if getattr(twcfg, "rho_min_weight", None) is not None else rho_min_w)
            aux_type = str(getattr(twcfg, "aux_loss_type", aux_type) or aux_type).lower()
            aux_layers_spec = str(getattr(twcfg, "aux_layers", aux_layers_spec) or "")
        else:
            use_aux = self.use_aux
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        inputs = dict(getattr(self.trainer, "_twyd_inputs", {}) or {})
        if not inputs:
            # Fallback: wrapper caches inputs in forward() before swift strips them.
            try:
                cached = getattr(m, "_twyd_last_inputs", None) or getattr(raw_model, "_twyd_last_inputs", None)
                if cached:
                    inputs = dict(cached)
            except Exception:
                pass
        t_logits, t_hidden = None, None
        get_t = getattr(m, "get_teacher_outputs", None)
        if get_t is None:
            get_t = getattr(raw_model, "get_teacher_outputs", None)
        if get_t is not None and inputs:
            try:
                _tt = _tick() if _p else 0.0
                # Teacher hidden states cost ~428 MB at SEQ=2048 (34 layers x 1x2048x3072
                # bf16) and OOM'd a 24 GB card; ask for logits only here and fetch the
                # hidden states from a second, truncated forward inside the aux block.
                t_out = get_t(output_hidden_states=False,
                              **{k: v for k, v in inputs.items()
                                 if k in ("input_ids", "attention_mask", "position_ids")})
                t_logits = (t_out.logits if hasattr(t_out, "logits") else t_out["logits"]).detach()
                if _p:
                    _PROF["t_teacher"] += _tock(_tt)
                t_hidden = None  # filled by the aux block via a short second forward
            except Exception:
                t_logits = None
        shift_s = logits[..., :-1, :]
        if shift_s.size(1) == 0:
            return logits.sum() * 0.0
        loss = 0.0
        ntp = None
        stage = self.stage
        if twcfg is not None:
            stage = str(getattr(twcfg, "stage", stage)).lower()
        is_sft = (stage == "sft")
        shift_y = labels[..., 1:] if labels is not None else None
        per_pos = None
        rho_w = None
        V_s = int(shift_s.size(-1))
        V_t = int(t_logits.size(-1)) if t_logits is not None else 0
        use_fused_topk = (
            t_logits is not None
            and 0 < int(kl_topk or 0) < min(V_s, V_t)
        )
        if use_fused_topk:
            _tk = _tick() if _p else 0.0
            use_rho = (not is_sft) and (0.0 < rho_keep < 1.0) and (shift_y is not None)
            shift_t = t_logits[..., :-1, :].detach()
            t_logits = None  # release outer reference early
            per_pos, ce_per_pos, rho_w = self._fused_topk_distill(
                shift_s, shift_t, shift_y, T, int(kl_topk),
                use_rho=use_rho, rho_keep=rho_keep, rho_min_w=rho_min_w,
            )
            del shift_t
            if _p:
                _PROF["t_kl"] += _tock(_tk)
            if shift_y is not None and ce_per_pos is not None:
                valid = (shift_y != -100).to(ce_per_pos.dtype)
                if is_sft:
                    ntp = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1.0)
                elif rho_w is not None:
                    w_eff = rho_w * valid
                    ntp = (ce_per_pos * w_eff).sum() / w_eff.sum().clamp_min(1.0)
                else:
                    ntp = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1.0)
                loss = loss + wNTP * ntp
        else:
            shift_t = None
            if t_logits is not None:
                _tk = _tick() if _p else 0.0
                shift_t = t_logits[..., :-1, :].detach()
                per_pos = self._per_pos_kl(shift_s, shift_t, T, topk=kl_topk)  # [B,S-1]
                if _p:
                    _PROF["t_kl"] += _tock(_tk)
            use_rho = (not is_sft) and (0.0 < rho_keep < 1.0) and (per_pos is not None)
            if shift_y is not None:
                _tc = _tick() if _p else 0.0
                if is_sft:
                    ntp = F.cross_entropy(shift_s.reshape(-1, V_s),
                                          shift_y.reshape(-1), ignore_index=-100)
                elif use_rho:
                    rho_w = self._rho_token_weights(shift_s, shift_t, shift_y, per_pos, T,
                                                    rho_keep, rho_min_w)
                    ce_per_pos = F.cross_entropy(
                        shift_s.reshape(-1, V_s), shift_y.reshape(-1), reduction="none"
                    ).view_as(rho_w)
                    ntp = (ce_per_pos * rho_w).sum() / rho_w.sum().clamp_min(1.0)
                else:  # pt: labels==input_ids, dense supervision
                    ntp = F.cross_entropy(shift_s.reshape(-1, V_s),
                                          shift_y.reshape(-1))
                if _p:
                    _PROF["t_ce"] += _tock(_tc)
                loss = loss + wNTP * ntp
        kl = None
        if per_pos is not None:
            if is_sft and shift_y is not None:
                kl = self._masked_mean(per_pos, shift_y, loss_scale)
            elif rho_w is not None:
                w_eff = rho_w * ((shift_y != -100).to(rho_w.dtype) if shift_y is not None else 1.0)
                kl = (per_pos * w_eff).sum() / w_eff.sum().clamp_min(1.0)
            elif shift_y is not None:
                valid = (shift_y != -100).to(per_pos.dtype)
                kl = (per_pos * valid).sum() / valid.sum().clamp_min(1.0)
            else:  # pt without labels: every position distills
                kl = per_pos.mean()
            loss = loss + wKL * kl
        aux = None
        if use_aux and lam > 0:
            try:
                # Alignment does not need the full context. Re-forwarding the student
                # over SEQ=2048 with output_hidden_states=True costs ~2 GiB of
                # activations, which OOM'd a 24 GB card at SEQ=2048 and was then
                # swallowed by the `except` below - i.e. the aux term silently
                # vanished for the whole 500M-token aux arm (bug 8, 2026-11-13).
                # Truncate the alignment window (default 512 tokens).
                _aux_seq = int((self.kw or {}).get("max_aux_seq", 512) or 512)
                _aux_inputs = {k: v for k, v in inputs.items()
                               if k in ("input_ids", "attention_mask", "position_ids")}
                if _aux_inputs.get("input_ids") is not None:
                    _full = inputs.get("input_ids").shape[1]
                    if _full > _aux_seq:
                        _aux_inputs = {k: v[:, :_aux_seq] for k, v in _aux_inputs.items()}
                # Second, short teacher forward for the alignment targets only.
                if get_t is not None and _aux_inputs:
                    try:
                        _t_short = get_t(output_hidden_states=True, **_aux_inputs)
                        t_hidden = getattr(_t_short, "hidden_states", None)
                    except Exception as _e:
                        t_hidden = None
                        if not getattr(self, "_aux_err_logged", False):
                            self._aux_err_logged = True
                            print(f"[twyd] L_align teacher hidden forward FAILED: "
                                  f"{type(_e).__name__}: {_e}", flush=True)
                s_hidden = getattr(outputs, "hidden_states", None)
                if s_hidden is None:
                    # Ask the wrapper to re-run student with hidden states only
                    # when the template did not request them (saves memory).
                    # Guard: synthetic test doubles (FakeBlock without forward)
                    # must not crash the loss — aux stays None there.
                    try:
                        has_real_forward = False
                        try:
                            st = getattr(m, "student", None)
                            has_real_forward = st is not None and not type(st).__name__.startswith("Fake")
                        except Exception:
                            has_real_forward = False
                        if _aux_inputs and has_real_forward:
                            # Must stay in the autograd graph: the aux term is a
                            # training loss. The old no_grad() here made L_align
                            # a constant even when teacher hidden states existed.
                            _s = m.student(**_aux_inputs, output_hidden_states=True,
                                           use_cache=False)
                            s_hidden = getattr(_s, "hidden_states", None)
                    except Exception as _e:
                        s_hidden = None
                        if not getattr(self, "_aux_err_logged", False):
                            self._aux_err_logged = True
                            print(f"[twyd] L_align student re-forward FAILED: "
                                  f"{type(_e).__name__}: {_e}", flush=True)
                if t_hidden is not None and s_hidden is not None:
                    _ls = s_hidden[0].shape[1]
                    if t_hidden[0].shape[1] != _ls:
                        t_hidden = tuple(t[:, :_ls] for t in t_hidden)
                if s_hidden is not None and t_hidden is not None and len(s_hidden) and len(t_hidden):
                    L = min(len(s_hidden), len(t_hidden))
                    max_aux_layers = int((self.kw or {}).get("max_aux_layers", 4) or 4)
                    if aux_layers_spec.strip():
                        from .config import parse_layer_range as _parse_lr
                        idxs = sorted(_parse_lr(aux_layers_spec, L))
                    else:
                        idxs = [round(i * (L - 1) / max(max_aux_layers - 1, 1))
                                for i in range(min(max_aux_layers, L))]
                    acc, cnt = 0.0, 0
                    P = None
                    if aux_type == "logit_lens":
                        st_mod = getattr(m, "student", m)
                        tc_mod = getattr(m, "teacher", None)
                        lens_topk = max(kl_topk, 32)
                        lens_T = min(max(T, 1.0), 4.0)
                        for li in idxs:
                            try:
                                sh, th = s_hidden[li], t_hidden[li]
                                # Subsample up to 128 tokens to keep mid-layer vocab projection tiny
                                step_tok = max(1, sh.shape[1] // 128)
                                sh_sub = sh[:, ::step_tok, :]
                                th_sub = th[:, ::step_tok, :].detach()
                                zs_mid = self._project_logit_lens(st_mod, sh_sub)
                                with torch.no_grad():
                                    zt_mid = self._project_logit_lens(tc_mod, th_sub) if tc_mod is not None else None
                                if zs_mid is None or zt_mid is None:
                                    continue
                                term = self._per_pos_kl(
                                    zs_mid, zt_mid, lens_T, chunk=64, topk=lens_topk
                                ).mean() / (lens_T * lens_T)
                                if torch.isfinite(term):
                                    acc = acc + term
                                    cnt += 1
                            except Exception:
                                continue
                    else:
                        # Eq.7 hidden part: 1/L Σ_l ||h^S_l P - h^T_l||^2, averaged
                        # over L matched layers (not just the last). P is one
                        # least-squares projection shared across layers (no params).
                        sh0, th0 = s_hidden[-1], t_hidden[-1]
                        if sh0.shape[-1] != th0.shape[-1]:
                            P = fit_hidden_projection(
                                sh0.reshape(-1, sh0.shape[-1]),
                                th0.reshape(-1, th0.shape[-1]),
                                int((self.kw or {}).get("max_aux_fit_rows", 1024) or 1024))
                        for li in idxs:
                            try:
                                sh, th = s_hidden[li], t_hidden[li]
                                pred = sh
                                if P is not None and sh.shape[-1] != th.shape[-1]:
                                    pred = sh @ P.to(device=sh.device, dtype=sh.dtype)
                                # Never let mse_loss broadcast a width mismatch: that
                                # turns L_align into a different (wrong) objective.
                                if pred.shape != th.shape:
                                    continue
                                if is_sft and shift_y is not None:
                                    pr = pred[..., :-1, :].contiguous()
                                    tg = th[..., :-1, :].contiguous()
                                    # shift_y spans the full training window; the aux
                                    # window may be truncated (max_aux_seq), so slice it.
                                    _sy = shift_y
                                    if _sy.shape[-1] != pr.shape[1] + 1:
                                        _sy = _sy[..., : pr.shape[1] + 1]
                                    mm = (_sy != -100).unsqueeze(-1).to(pr.dtype)
                                    term = (((pr - tg.float()) ** 2) * mm).sum() / mm.sum().clamp_min(1.0)
                                else:
                                    term = F.mse_loss(pred.float(), th.float())
                                if self._aux_norm:
                                    # Relative alignment error: ||h^S P - h^T||^2 / ||h^T||^2.
                                    _pf = pred.float()
                                    _tf = th.float()
                                    _num = ((_pf - _tf) ** 2).sum(-1)
                                    _den = (_tf ** 2).sum(-1).clamp_min(1e-6)
                                    term = (_num / _den).mean()
                                if torch.isfinite(term):
                                    acc = acc + term
                                    cnt += 1
                            except Exception:
                                continue
                    if cnt > 0:
                        aux = acc / cnt
                        if not getattr(self, "_aux_logged", False):
                            self._aux_logged = True
                            _av = float(aux.detach()) if torch.is_tensor(aux) else float(aux)
                            print(f"[twyd] L_align active ({aux_type}): layers={idxs} "
                                  f"P={None if P is None else tuple(P.shape)} "
                                  f"aux={_av:.4g} lam={lam} "
                                  f"lam*aux={lam * _av:.4g} norm={self._aux_norm}", flush=True)
                    elif not getattr(self, "_aux_skip_logged", False):
                        # Silent drops are how the old aux term went missing for
                        # a whole study: say WHY instead of skipping quietly.
                        self._aux_skip_logged = True
                        _sd = None if s_hidden is None else (len(s_hidden), tuple(s_hidden[-1].shape))
                        _td = None if t_hidden is None else (len(t_hidden), tuple(t_hidden[-1].shape))
                        print(f"[twyd] L_align SKIPPED: s_hidden={_sd} t_hidden={_td} "
                              f"P={None if P is None else tuple(P.shape)} idxs={idxs} "
                              f"use_aux={use_aux} lam={lam}", flush=True)
            except Exception as _e:
                aux = None
                if not getattr(self, "_aux_err_logged", False):
                    self._aux_err_logged = True
                    print(f"[twyd] L_align DISABLED this run: {type(_e).__name__}: {_e}",
                          flush=True)
            if aux is not None and torch.isfinite(aux):
                loss = loss + lam * aux
        try:
            self.trainer._twyd_last = {"kl": None if kl is None else float(kl.detach()),
                                       "ntp": None if ntp is None else float(ntp.detach()),
                                       "aux": None if aux is None else float(aux.detach())}
        except Exception:
            pass
        if loss_scale is not None and not torch.is_tensor(loss_scale):
            # scalar multiplier only; tensor loss_scale already consumed
            # per-term (SFT masked KL) above to avoid double counting.
            try:
                loss = loss * float(loss_scale)
            except Exception:
                pass
        if _p:
            _PROF["n"] += 1
            _PROF["t_fwd"] += _tock(_t0)
            n = _PROF["n"]
            if n % 5 == 0:
                print(f"[twyd-prof] calls={n} "
                      f"teacher={_PROF['t_teacher'] / n * 1e3:.0f}ms "
                      f"kl={_PROF['t_kl'] / n * 1e3:.0f}ms "
                      f"ce={_PROF['t_ce'] / n * 1e3:.0f}ms "
                      f"loss_total={_PROF['t_fwd'] / n * 1e3:.0f}ms "
                      f"(teacher+kl+ce are inside loss_total)", flush=True)
        # Component logging: the number the trainer prints is CE + T^2*KL, which is
        # dominated by KL. Print both so "loss looks flat" can be read correctly.
        _CALLS["n"] += 1
        every = int(os.environ.get("TWYD_LOG_EVERY", 320) or 320)
        if every > 0 and _CALLS["n"] % every == 0 and ntp is not None:
            try:
                _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            except Exception:
                _rank = 0
            if _rank == 0:
                _kl = float(kl.detach()) if torch.is_tensor(kl) else (float(kl) if kl is not None else float("nan"))
                _ntp = float(ntp.detach()) if torch.is_tensor(ntp) else float(ntp)
                _ls = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
                # Report MoE router balance diagnostics when TwydSharedRoutedMoE layers exist.
                _moe_str = ""
                try:
                    _st = getattr(m, "student", m)
                    _loads, _lbs = [], []
                    for _mod in _st.modules():
                        if hasattr(_mod, "last_load") and _mod.last_load is not None:
                            _loads.append(_mod.last_load)
                            _lbs.append(float(getattr(_mod, "last_aux_val", 0.0)))
                    if _loads:
                        _avg_load = torch.stack(_loads, dim=0).mean(dim=0)
                        _moe_str = (f" moe_aux={sum(_lbs)/len(_lbs):.4f} "
                                    f"load=[{float(_avg_load.min()):.3f}..{float(_avg_load.max()):.3f}]")
                except Exception:
                    pass
                # `kl` already carries the T^2 factor from _per_pos_kl, so do NOT
                # multiply again; report the raw nats too for interpretability.
                print(f"[twyd] ce={_ntp:.4f} kl_raw={_kl / (T * T):.6f} "
                      f"klT2={wKL * _kl:.2f} loss={_ls:.2f}{_moe_str}", flush=True)
        return loss
