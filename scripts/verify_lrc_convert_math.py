"""Verify LRC->twyD conversion math on small synthetic weights (no model loading)."""
import torch
from twyd.tuner import CoreLinear

torch.manual_seed(0)
d_ff, H, r = 64, 24, 12

# Teacher weight (d_ff x H)
t_w = torch.randn(d_ff, H)
T, U_perp = torch.linalg.svd(t_w.float(), full_matrices=True)[0][:, :H].contiguous(), \
            torch.linalg.svd(t_w.float(), full_matrices=True)[0][:, H:].contiguous()

# LRC-style W_lrc (d_ff x r) from teacher slice
W_lrc = (t_w[:, :r]).contiguous()

# twyd conversion: Z_col = (T^T @ W_lrc).T  =>  (r x H)
Z_col = (T.T @ W_lrc).T.contiguous()

# Build twyd CoreLinear (plain: with_perp=False) and check deployed W matches projection
plain = CoreLinear(T, Z_col, side="left", with_perp=False)
W_deployed = plain.merged_weight()  # (d_ff x r)
# T T^T is the projector onto teacher col-space; W_lrc sliced from teacher is already in it,
# so T T^T @ W_lrc should equal W_lrc (up to float error).
proj = T @ T.T @ W_lrc
print("gate/up: deployed W == projector @ W_lrc:", torch.allclose(W_deployed, proj, atol=1e-4))
print("         max diff:", float((W_deployed - proj).abs().max()))

# Core with Z_perp=0 should be identical to plain
core = CoreLinear(T, Z_col, U_perp, torch.zeros(r, d_ff - H), side="left", with_perp=True)
print("core (Z_perp=0) == plain:", torch.allclose(core.merged_weight(), plain.merged_weight(), atol=1e-6))

# Down (right): W_lrc = (r x d_ff), R = Vh[:H, :], Z_col = W_lrc @ R.T  => (r x H)
t_w_down = torch.randn(H, d_ff)
_, _, Vh = torch.linalg.svd(t_w_down.float(), full_matrices=True)
R = Vh[:H, :].contiguous()
V_perp = Vh[H:, :].contiguous()
W_lrc_down = t_w_down[:r, :].contiguous()  # (r x d_ff)
Z_col_down = (W_lrc_down @ R.T).contiguous()  # (r x H)
core_down = CoreLinear(R, Z_col_down, V_perp, torch.zeros(r, d_ff - H), side="right", with_perp=True)
W_down = core_down.merged_weight()  # (r x d_ff)
proj_down = W_lrc_down @ R.T @ R  # project row-space
print("down:  deployed W == projector-row @ W_lrc:", torch.allclose(W_down, proj_down, atol=1e-4))
print("         max diff:", float((W_down - proj_down).abs().max()))
print("ALL MATH CHECKS DONE")
