"""ms-swift loss_map plugin: lrc_distill.

L = wKL * T^2 * KL(softmax(zT/T) || softmax(zS/T))
  + wNTP * CE(y, zS)
  + lambda_aux * L_align   (hidden MSE, Eq.7 minimal; attn/MLP alignment off by default)

Teacher forward runs under no_grad via wrapper.get_teacher_outputs(); inputs
are read from (1) trainer._twyd_inputs (callback), (2) wrapper._twyd_last_inputs
(cached in _TwydWrapper.forward, survives ms-swift input stripping).
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
        self.T = float(kw.get("kl_temperature", 40.0))
        self.wKL = float(kw.get("kl_weight", 1.0))
        self.wNTP = float(kw.get("ntp_weight", 1.0))
        self.lam = float(kw.get("aux_weight", 0.2))
        self.use_aux = bool(kw.get("use_aux", True))

    def _unwrap(self, model):
        m = model
        for _ in range(5):
            inner = getattr(m, "model", None)
            if inner is None or inner is m:
                break
            m = inner() if callable(inner) else inner
        return m

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
        if labels is not None:
            shift_y = labels[..., 1:].contiguous()
            ntp = F.cross_entropy(shift_s.view(-1, shift_s.size(-1)),
                                  shift_y.view(-1), ignore_index=-100)
            loss = loss + wNTP * ntp
        kl = None
        if t_logits is not None:
            shift_t = t_logits[..., :-1, :].contiguous().detach()
            V = min(shift_s.size(-1), shift_t.size(-1))
            kl = F.kl_div(F.log_softmax(shift_s[..., :V] / T, dim=-1),
                          F.softmax(shift_t[..., :V] / T, dim=-1),
                          reduction="batchmean") * (T * T)
            loss = loss + wKL * kl
        aux = None
        if use_aux and lam > 0:
            # Minimal Eq.7: last-hidden MSE with teacher-width projection.
            # Full per-layer hidden/attn/MLP alignment stays future work; previously
            # this term was exactly zero (aux = zeros), now it measures something.
            try:
                s_hidden = getattr(outputs, "hidden_states", None)
                if s_hidden is not None and t_hidden is not None and len(s_hidden) and len(t_hidden):
                    sh, th = s_hidden[-1], t_hidden[-1]
                    if sh.shape[-1] != th.shape[-1]:
                        # Reuse teacher-col basis idea: least-squares project student->teacher width.
                        # Cheap per-batch solve; no new parameters (not stored).
                        A, B = sh.detach().reshape(-1, sh.shape[-1]), th.detach().reshape(-1, th.shape[-1])
                        P = torch.linalg.lstsq(A.float(), B.float()).solution.to(sh.dtype)  # r x H
                        aux = F.mse_loss(sh @ P, th)
                    else:
                        aux = F.mse_loss(sh.float(), th.float())
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
        if loss_scale is not None:
            loss = loss * loss_scale
        return loss
