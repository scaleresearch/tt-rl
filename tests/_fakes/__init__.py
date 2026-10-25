"""Shared test fakes. Not a test module; importing it collects nothing.

Re-exports are lazy on purpose: `nets` and `mesh` reach into `ttrl.engine`,
and `chips` enumerates the box's chips at import time, so a test that only
wants `tensors` must not pay for either.
"""

from __future__ import annotations

import importlib

_HOMES = {
    "tensors": ("FakeTensor", "FakeModule", "FakeAdam", "TapeData", "MatmulData"),
    "nets": ("upload_obs", "assert_every_parameter_trains"),
    "reports": ("perf_report", "throughput_report"),
    "mesh": ("patch_upload", "exact_sum_reduce"),
    "chips": (
        "two_candidate_chips",
        "fuser_free",
        "TWO_CHIPS",
        "TWO_IDLE_CHIPS_AVAILABLE",
        "requires_two_idle_chips",
        "two_idle_chips_gate",
    ),
}

_OWNER = {name: mod for mod, names in _HOMES.items() for name in names}

__all__ = sorted(_OWNER)


def __getattr__(name):
    try:
        home = _OWNER[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f"{__name__}.{home}"), name)
    globals()[name] = value
    return value


def __dir__():
    return __all__
