"""Memory-flatness helpers for the hardware soak guards. Test-only.

Two instruments, one trend fit. `dram_allocated_bytes` reads the device
allocator and `host_rss_bytes` reads this process's resident set; both feed
`measure`, `growth_bytes_per_step` and `assert_no_growth`, which care only
about a series of byte counts.

These back the four hardware DRAM-flatness guards that are the regression gate
for the cyclic-tape leak that killed PPO at ~47k steps, at ~485 MB per update.
They live in tests/ because they are test infrastructure, never shipped.

A leak of DEVICE memory is invisible to every Python-side tool: getrefcount,
tracemalloc and gc all measure host bytes, and a ttnn.Tensor is a tiny host
object holding a large device buffer. The only honest instrument is the device
allocator itself.

The leak this was written for does not grow smoothly. It grows in a sawtooth,
climbing for tens of updates and dropping back when CPython's generational
collector happens to run a gen-2 pass, so sampling two points and subtracting
calls a leaking run flat about half the time. measure_dram takes a series and
assert_no_growth fits a trend across it.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

import ttnn

__all__ = ["dram_allocated_bytes", "host_rss_bytes", "measure_dram",
           "growth_bytes_per_step", "assert_no_growth"]


def host_rss_bytes() -> int:
    """This process's resident set size in bytes, from /proc/self/statm.

    RSS rather than tracemalloc or gc: the host growth the README reports is
    what the operating system charges the process, which includes the arena
    fragmentation and the numpy buffers a pure-Python accounting misses. The
    second field of statm is resident pages.
    """
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def dram_allocated_bytes(device) -> int:
    """Total DRAM bytes currently allocated on `device`, across all banks.

    total_bytes_allocated_per_bank and num_banks are plain attributes, not
    methods: calling them raises "TypeError: 'int' object is not callable".
    Their product is the number the allocator reports in its own out-of-memory
    message, so a series is directly comparable against the "allocated: N B"
    in a "Not enough space to allocate ... across 8 banks" TT_FATAL.
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


def measure_dram(
    device,
    step: Callable[[int], object],
    *,
    steps: int,
    warmup: int = 0,
    on_sample: Callable[[int, int], None] | None = None,
) -> list[int]:
    """Run `step(i)` `warmup + steps` times, returning DRAM bytes after each of
    the last `steps` calls.

    `warmup` covers the allocations that legitimately happen once and keep
    memory: ttnn's program cache filling, conv2d materialising its prepared
    weight scratch, an optimizer allocating its moments on first use. Raise it
    for a bigger model rather than loosening the growth tolerance, which would
    blind the check to a real slow leak.

    `step`'s return value is discarded deliberately: holding it here would keep
    the last graph alive and put a one-tape offset into the series.
    """
    for i in range(warmup):
        step(i)
    series: list[int] = []
    for i in range(steps):
        step(warmup + i)
        sample = dram_allocated_bytes(device)
        series.append(sample)
        if on_sample is not None:
            on_sample(warmup + i, sample)
    return series


def growth_bytes_per_step(series: Sequence[int]) -> float:
    """Least-squares slope of `series` in bytes per step.

    A first-vs-last difference is not enough: the sawtooth can put the last
    sample below the first in a run that is unambiguously leaking.
    """
    n = len(series)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(series) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(series))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return 0.0 if den == 0 else num / den


def assert_no_growth(
    series: Sequence[int],
    *,
    max_bytes_per_step: float = 0.0,
    window: int = 8,
    tolerance_bytes: int = 0,
    label: str = "device DRAM",
) -> None:
    """Raise AssertionError unless `series` is flat.

    Two checks, because each alone has a blind spot. Trend: the least-squares
    slope must not exceed max_bytes_per_step, which catches a slow leak whose
    per-step increment is below the sampling noise. Level: the mean of the last
    `window` samples must not exceed the mean of the first `window` by more
    than tolerance_bytes, which catches a step change that a slope fit over a
    long flat run would dilute below threshold.

    The defaults are exact equality because on a correct build this series is
    bit-identical from sample to sample: the allocator hands back the same
    addresses for the same sequence of same-shaped allocations.
    """
    if len(series) < 2:
        raise AssertionError(f"{label}: need at least 2 samples, got {len(series)}")

    slope = growth_bytes_per_step(series)
    detail = (
        f"{label}: first={series[0] / 1e6:.2f} MB last={series[-1] / 1e6:.2f} MB "
        f"min={min(series) / 1e6:.2f} MB max={max(series) / 1e6:.2f} MB "
        f"slope={slope / 1e6:.4f} MB/step over {len(series)} samples"
    )
    if slope > max_bytes_per_step:
        raise AssertionError(
            f"{label} grows across the run ({slope / 1e6:.4f} MB/step > "
            f"{max_bytes_per_step / 1e6:.4f} MB/step allowed). {detail}"
        )

    w = min(window, len(series) // 2)
    if w >= 1:
        head = sum(series[:w]) / w
        tail = sum(series[-w:]) / w
        if tail - head > tolerance_bytes:
            raise AssertionError(
                f"{label} ends higher than it started "
                f"(last-{w} mean {tail / 1e6:.2f} MB vs first-{w} mean "
                f"{head / 1e6:.2f} MB, > {tolerance_bytes / 1e6:.2f} MB allowed). "
                f"{detail}"
            )
