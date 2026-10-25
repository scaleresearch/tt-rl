"""Precision policy: the one dtype the engine keeps parameters and state in.

The engine runs fp32 end to end. Forward and backward both stay fp32, with
`ops.matmul` and `conv.config` pinning HiFi4 + `fp32_dest_acc_en` as this
hardware's accuracy ceiling, so nothing casts down and there is no second
compute dtype.

The fused `*_bw` gradient kernels do accept and return fp32 on ttnn 0.80.0.
Measured on Blackhole against a torch reference, `relu_bw`/`tanh_bw`/
`sigmoid_bw`/`exp_bw`/`sqrt_bw` return FLOAT32 with max abs error 0.0,
2.09e-07, 1.64e-07, 9.54e-07 and 1.91e-06; `clip_bw` and `clamp_bw` are
exact. The 0.80 docstrings still list BFLOAT16 only, so a ttnn bump wants
re-measuring rather than re-reading them.

`MASTER_DTYPE` is the invariant `optim.adam` enforces on every parameter it
is handed: optimizer state, losses, TD targets, advantages and log-probs all
stay fp32. It is imported there, so this constant is the single place the
invariant is declared.

bfloat8_b is deliberately unnamed here and must stay that way for gradients and
optimizer state.
"""

from __future__ import annotations

import ttnn

MASTER_DTYPE = ttnn.float32
