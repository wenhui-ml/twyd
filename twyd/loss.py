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
        self.stage = str(kw.get("stage", "pt")).lower()

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

    def _per_pos_kl(self, shift_s, shift_t, T, chunk=256):
        """Per-position KL(t||s) in fp32, chunked over the sequence.

        The distillation temperature makes this numerically hostile in bf16:
        with T=40 the logits are divided down to ~1e-2, while bf16 resolution
        around log p ~ -log(V) ~ -11.8 (V=128k) is ~5e-2.  So the log-probs
        carry rounding noise LARGER than the signal, and the ``T*T`` rescaling
        (1600x) amplifies that noise into the loss.  On a real 1.5B run this
        produced a *negative* training loss (-3.4) that drifted further
        negative every step while token_acc looked healthy.

        Fix: do the softmax/kl math in fp32 and chunk the sequence so the
        [B, S, V] fp32 temporaries stay small.  KL >= 0 is then guaranteed.
        """
        V = min(shift_s.size(-1), shift_t.size(-1))
        outs = []
        for s_c, t_c in zip(shift_s[..., :V].split(chunk, dim=1),
                            shift_t[..., :V].split(chunk, dim=1)):
            logp_s = F.log_softmax(s_c.float() / T, dim=-1)
            logp_t = F.log_softmax(t_c.float() / T, dim=-1)
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

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        raw_model = self.trainer.model
        _p = _prof_on()
        _t0 = _tick() if _p else 0.0
        m = self._unwrap(raw_model)
        twcfg = getattr(m, "twyd_cfg", None) or getattr(raw_model, "twyd_cfg", None)
        T, wKL, wNTP, lam = self.T, self.wKL, self.wNTP, self.lam
        if twcfg is not None:
            T, wKL, wNTP, lam = twcfg.kl_temperature, twcfg.kl_weight, twcfg.ntp_weight, twcfg.aux_weight
            use_aux = twcfg.use_aux
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
                _want_hidden = bool(use_aux and lam > 0)
                _tt = _tick() if _p else 0.0
                t_out = get_t(output_hidden_states=_want_hidden,
                              **{k: v for k, v in inputs.items()
                                 if k in ("input_ids", "attention_mask", "position_ids")})
                t_logits = (t_out.logits if hasattr(t_out, "logits") else t_out["logits"]).detach()
                if _p:
                    _PROF["t_teacher"] += _tock(_tt)
                try:
                    # hidden_states only if model was run with output_hidden_states=True
                    t_hidden = getattr(t_out, "hidden_states", None)
                except Exception:
                    t_hidden = None
            except Exception:
                t_logits = None
        shift_s = logits[..., :-1, :].contiguous()
        loss = 0.0
        ntp = None
        stage = self.stage
        if twcfg is not None:
            stage = str(getattr(twcfg, "stage", stage)).lower()
        is_sft = (stage == "sft")
        shift_y = None
        if labels is not None:
            shift_y = labels[..., 1:].contiguous()
            _tc = _tick() if _p else 0.0
            if is_sft:
                ntp = F.cross_entropy(shift_s.view(-1, shift_s.size(-1)),
                                      shift_y.view(-1), ignore_index=-100)
            else:  # pt: labels==input_ids, dense supervision
                ntp = F.cross_entropy(shift_s.view(-1, shift_s.size(-1)),
                                      shift_y.view(-1))
            if _p:
                _PROF["t_ce"] += _tock(_tc)
            loss = loss + wNTP * ntp
        kl = None
        if t_logits is not None:
            _tk = _tick() if _p else 0.0
            shift_t = t_logits[..., :-1, :].contiguous().detach()
            per_pos = self._per_pos_kl(shift_s, shift_t, T)  # [B,S-1]
            if _p:
                _PROF["t_kl"] += _tock(_tk)
            if is_sft and shift_y is not None:
                kl = self._masked_mean(per_pos, shift_y, loss_scale)
            else:  # pt: every position distills
                kl = per_pos.mean()
            loss = loss + wKL * kl
        aux = None
        if use_aux and lam > 0:
            try:
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
                        if inputs and has_real_forward:
                            # Must stay in the autograd graph: the aux term is a
                            # training loss. The old no_grad() here made L_align
                            # a constant even when teacher hidden states existed.
                            _s = m.student(**{k: v for k, v in inputs.items()
                                              if k in ("input_ids", "attention_mask", "position_ids")},
                                           output_hidden_states=True, use_cache=False)
                            s_hidden = getattr(_s, "hidden_states", None)
                    except Exception:
                        s_hidden = None
                if s_hidden is not None and t_hidden is not None and len(s_hidden) and len(t_hidden):
                    # Eq.7 hidden part: 1/L Σ_l ||h^S_l P - h^T_l||^2, averaged
                    # over L matched layers (not just the last). P is one
                    # least-squares projection shared across layers (no params).
                    L = min(len(s_hidden), len(t_hidden))
                    # Cap layers to bound the lstsq cost on long contexts.
                    max_aux_layers = int((self.kw or {}).get("max_aux_layers", 4) or 4)
                    idxs = [round(i * (L - 1) / max(max_aux_layers - 1, 1)) for i in range(min(max_aux_layers, L))]
                    acc, cnt = 0.0, 0
                    # Fit P once on the last layer, reuse for all (cheap + stable).
                    sh0, th0 = s_hidden[-1], t_hidden[-1]
                    P = None
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
                                mm = (shift_y != -100).unsqueeze(-1).to(pr.dtype)
                                term = (((pr - tg.float()) ** 2) * mm).sum() / mm.sum().clamp_min(1.0)
                            else:
                                term = F.mse_loss(pred.float(), th.float())
                            if torch.isfinite(term):
                                acc = acc + term
                                cnt += 1
                        except Exception:
                            continue
                    if cnt > 0:
                        aux = acc / cnt
                        if not getattr(self, "_aux_logged", False):
                            self._aux_logged = True
                            print(f"[twyd] L_align active: layers={idxs} "
                                  f"P={None if P is None else tuple(P.shape)}", flush=True)
            except Exception:
                aux = None
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
        return loss
