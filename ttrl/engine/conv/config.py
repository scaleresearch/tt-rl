"""ttnn placement and fidelity configuration for the six convolutions.

Forward, dX (conv_transpose2d) and the dW matmul all run through the configs
built here, so fidelity is stated once rather than inherited from a ttnn
default that can change between builds.
"""

from __future__ import annotations

import ttnn

from ttrl.engine.device import register_device_cache

__all__ = [
    "FP32", "HEIGHT_SHARDED", "accurate_compute_config", "conv_config",
    "is_shard_too_small", "rm", "tile",
]

FP32 = ttnn.float32
HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

#: HiFi4 + fp32_dest_acc_en, memoised per device. fp32 matmul on this hardware
#: runs as repeated bf16-granularity passes, so HiFi4 is the accuracy ceiling
#: (~1.3e-3 relative), not a performance knob.
_COMPUTE_CFG: dict = {}

#: Substrings identifying "this shard layout does not fit at this shape" among
#: the RuntimeErrors ttnn.conv2d and ttnn.conv_transpose2d raise. Matched on the
#: message because ttnn raises a bare RuntimeError for everything, TT_FATAL text
#: included. Still current in 0.80, and byte-identical to 0.77 for the first
#: three TT_FATALs. The L1 queries that are bound,
#: `get_max_worker_l1_unreserved_size` and `get_l1_alignment`, report device
#: wide capacity. Whether one conv's circular buffers fit at a given shard
#: layout is known only to the dispatch itself, so the placement search has to
#: attempt it and read the refusal.
_SHARD_DOES_NOT_FIT = (
    "found_valid_config",                        # op_slicing.cpp: DRAM auto-slice gave up
    "could not find valid slice",                # its human-readable half
    "num_shards.value() <= num_compute_banks",   # bank_manager.cpp
    # conv_transpose2d at a large batch: the height-sharded placement's
    # circular buffers do not fit, reported as two different TT_THROWs from
    # validate_circular_buffer_region depending on whether they overrun L1
    # outright or collide with the sharded tensor already resident in it.
    # Measured on Blackhole with ttnn 0.80 at conv2's geometry, batch 256 and
    # batch 64 respectively.
    "beyond max L1 size",                        # program.cpp
    "clash with L1 buffers",                     # program.cpp
)


def accurate_compute_config(device):
    cfg = _COMPUTE_CFG.get(id(device))
    if cfg is None:
        cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        _COMPUTE_CFG[id(device)] = cfg
    return cfg


def _purge(device) -> None:
    _COMPUTE_CFG.pop(id(device), None)


# id() is unique only among live objects, so a config keyed by a closed
# device's address would be served to the next device opened at that address.
register_device_cache(_purge)


def rm(t):
    """Row-major view. conv2d consumes activations row-major."""
    return t if t.get_layout() == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)


def tile(t):
    return t if t.get_layout() == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)


# USE_CONV_PRESHARD stays off, and the pre-sharded input path is therefore not
# implemented here: turning it on fails the L1 budget on several gate tests.


def is_shard_too_small(exc: BaseException) -> bool:
    return any(s in str(exc) for s in _SHARD_DOES_NOT_FIT)


# ---------------------------------------------------------------------------
# act_block_h_override: DELETED, and this is the record of why not
# ---------------------------------------------------------------------------
# A Conv2dConfig act_block_h_override of 64, applied at conv2's forward
# geometry, used to be carried here as a correctness pin: ~6.4% of end-to-end
# DQN update trials diverged on trace replay without it (Blackhole, ttnn 0.77).
# It is gone, on the strength of the measurement below. ARCH's rule is that the
# comment is the only record of why a workaround exists; this one is the record
# of why it does not.
#
# MEASURED on ttnn 0.80, Blackhole, 110-core grid. A full captured DQN update
# (online and target towers, huber loss, forward and backward in one trace),
# 128 replays at fixed weights plus 100 full updates stepped against an eager
# twin, per configuration. For each configuration the value ttnn SELECTED was
# logged, not the value requested, which is the whole point:
#
#   requested 0,  batch 32: 0/128 replays, 0/100 updates diverged
#   requested 64, batch 32: ttnn selected 32 (logged 216 times); 0/128, 0/100
#   requested 32, batch 32: applied as asked;                    0/128, 0/100
#   requested 0,  batch 64: 0/128, 0/100
#   requested 64, batch 64: applied as asked;                    0/128, 0/100
#   16 independent seeded processes, requested 0, batch 32: one distinct loss
#   and gradient fingerprint across all 16, so capture-to-capture is
#   deterministic as well as replay-to-replay
#
# Two facts came out of it. The requested 64 never ran at the geometry the pin
# was written for: 64 rows is two tiles, that conv has one output tile per
# core, and ttnn silently selects 32 (it selects 32 at batch 128 too, where 3
# and 15 tiles per core are not divisible by 2, and applies 64 as asked at
# batch 64, where there are exactly 2). So the shipped field documented a value
# the hardware never saw, and the historical 6.4% was measured against an
# effective 32. And at an effective 0, 32 or 64 the rate on 0.80 is 0%.
#
# WHAT THIS DOES NOT PROVE. ttnn 0.77, where the 6.4% was measured, is not
# installed on this host and could not be tested, so the experiment cannot
# separate "upstream fixed it quietly between 0.77 and 0.80" from "the override
# never applied in the first place". The deletion is therefore scoped to the
# pinned range (ttnn>=0.80,<0.81) and guarded by
# tests/hardware/test_conv_trace_determinism.py. If divergence ever reappears
# on a future ttnn, start here: re-run that test, log what ttnn SELECTS rather
# than what is requested, and treat a per-core output tile count that does not
# divide the requested block height as the first suspect.


def conv_config(shard_layout, *, weights_dtype=FP32) -> "ttnn.Conv2dConfig":
    """The Conv2dConfig a conv and its weight preparation must share.

    `weights_dtype` is pinned to the master weight's dtype: conv2d reads it to
    decide whether the weight it was handed needs converting, and a mismatch
    silently re-derives the weight on host.

    act_block_h_override is deliberately not set; see the record above.
    enable_kernel_stride_folding is deliberately not set: at conv1's geometry
    it either raises or returns silently wrong output (pcc nan / 0.0021 /
    0.0030, measured). enable_activation_reuse is deliberately not set: it
    hangs with height sharding.
    """
    return ttnn.Conv2dConfig(weights_dtype=weights_dtype, shard_layout=shard_layout)
