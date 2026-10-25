"""EngineConfig: every value that changes engine results, stated explicitly."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EngineConfig", "DEFAULT"]


@dataclass(frozen=True)
class EngineConfig:
    # Whether execute_trace waits for replay to finish.
    trace_blocking: bool = False
    # conv2d allocates from the L1 "small" pool, which the allocator sizes
    # only when asked at open time. 32768 is the smallest value under which
    # every convolution in this repo opens without the allocator refusing.
    l1_small_size: int = 32768
    # Bytes reserved at open time for captured command traces. 32 MiB holds
    # the traced forward and backward of the largest net in this repo.
    trace_region_size: int = 32 * 1024 * 1024


DEFAULT = EngineConfig()
