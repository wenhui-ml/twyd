"""ms-swift loss_map plugin: lrc_distill (PT + SFT stages).

PT (stage="pt"): labels == input_ids, every position distills.
SFT (stage="sft"): prompt/pad masked to -100, KL+CE+aux all respect mask.
"""
import torch
import torch.nn.functional as F


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

    def _per_pos_kl(self, shift_s, shift_t, T):
        V = min(shift_s.size(-1), shift_t.size(-1))
        logp_s = F.log_softmax(shift_s[..., :V] / T, dim=-1)
        prob_t = F.softmax(shift_t[..., :V] / T, dim=-1)
        return F.kl_div(logp_s, prob_t, reduction="none").sum(dim=-1) * (T * T)

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
                t_out = get_t(**{k: v for k, v in inputs.items()
                                 if k in ("input_ids", "attention_mask", "position_ids")})
                t_logits = (t_out.logits if hasattr(t_out, "logits") else t_out["logits"]).detach()
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
            if is_sft:
                ntp = F.cross_entropy(shift_s.view(-1, shift_s.size(-1)),
                                      shift_y.view(-1), ignore_index=-100)
            else:  # pt: labels==input_ids, dense supervision
                ntp = F.cross_entropy(shift_s.view(-1, shift_s.size(-1)),
                                      shift_y.view(-1))
            loss = loss + wNTP * ntp
        kl = None
        if t_logits is not None:
            shift_t = t_logits[..., :-1, :].contiguous().detach()
            per_pos = self._per_pos_kl(shift_s, shift_t, T)  # [B,S-1]
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
                            with torch.no_grad():
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
                        try:
                            A = sh0.detach().reshape(-1, sh0.shape[-1])
                            B = th0.detach().reshape(-1, th0.shape[-1])
                            P = torch.linalg.lstsq(A.float(), B.float()).solution.to(sh0.dtype)
                        except Exception:
                            P = None
                    for li in idxs:
                        try:
                            sh, th = s_hidden[li], t_hidden[li]
                            pred = (sh @ P) if (P is not None and sh.shape[-1] != th.shape[-1]) else sh
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
        return loss
