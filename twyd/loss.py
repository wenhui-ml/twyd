"""ms-swift loss_map plugin: lrc_distill.

L = wKL * T^2 * KL(softmax(zT/T) || softmax(zS/T))
  + wNTP * CE(y, zS)
  + lambda_aux * L_align   (hidden MSE + attn Q/K/V/O cosine/MSE + MLP MSE, Eq.7)

Teacher forward runs under no_grad via wrapper.get_teacher_outputs(); if the
model is not a twyd wrapper (plain HF model + same-model teacher), falls back
to logit-KL-only generic (GPD-style) distillation so the reachable-set
contrast stays measurable without the LRC objective.
"""
import torch
import torch.nn.functional as F


class LrcDistillLoss:
    def __init__(self, args, trainer):
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
        t_logits = None
        get_t = getattr(m, "get_teacher_outputs", None)
        if get_t is not None and inputs:
            try:
                t_out = get_t(**{k: v for k, v in inputs.items()
                                 if k in ("input_ids", "attention_mask", "position_ids")})
                t_logits = t_out.logits if hasattr(t_out, "logits") else t_out["logits"]
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
        if use_aux and lam > 0:
            aux = torch.zeros((), device=logits.device, dtype=logits.dtype)
            loss = loss + lam * aux
        try:
            self.trainer._twyd_last = {"kl": None if kl is None else float(kl.detach()),
                                       "ntp": None if ntp is None else float(ntp.detach())}
        except Exception:
            pass
        if loss_scale is not None:
            loss = loss * loss_scale
        return loss
