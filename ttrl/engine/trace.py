"""ttnn command-trace capture and replay.

Capture is a callback API rather than a context manager on purpose. An
exception raised between `begin_trace_capture` and `end_trace_capture` leaves
the device in the recording state and wedges the chip, holding it until the
process is killed with SIGKILL. A callback puts every path out of the
recording region inside this module's own try, so no caller can open a
capture it fails to close.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import ttnn

from ttrl.engine.config import DEFAULT, EngineConfig

__all__ = [
    "HOST_READ_DURING_CAPTURE",
    "CapturedTrace",
    "TraceCaptureError",
    "capture",
]

#: ttnn's own error text when something inside a capture reads to host. Still
#: present in 0.80's libtt_metal.so, and 0.80 only adds capture restrictions,
#: so this string match is the only way to tell this failure from any other.
HOST_READ_DURING_CAPTURE = "Reads are not supported during trace capture."

R = TypeVar("R")


class TraceCaptureError(RuntimeError):
    """Trace capture failed. There is no eager fallback: a step that cannot be
    captured raises, rather than silently spending the run in eager mode."""


def _abort_capture(device, trace_id) -> None:
    """Leave the recording state after a failed capture.

    Load-bearing: without it the chip stays wedged. The device may already be
    out of the recording state, since the failure itself may have unwound it,
    so a second failure here is dropped rather than masking the original one.
    """
    try:
        ttnn.end_trace_capture(device, trace_id)
    except Exception:
        pass
    try:
        ttnn.release_trace(device, trace_id)
    except Exception:
        pass


class CapturedTrace:
    """One captured trace, replayable until closed."""

    __slots__ = ("device", "_trace_id")

    def __init__(self, device, trace_id):
        self.device = device
        self._trace_id = trace_id

    @property
    def live(self) -> bool:
        return self._trace_id is not None

    def replay(self, *, cfg: EngineConfig = DEFAULT, blocking: bool | None = None) -> None:
        """Execute the recorded commands again.

        Non-blocking by default: the replay is ordered ahead of any later
        device-to-host read on the same queue, so a caller that reads a metric
        afterwards is already synchronised, and the host is free meanwhile.
        """
        if self._trace_id is None:
            raise RuntimeError("replay() on a closed CapturedTrace")
        wait = cfg.trace_blocking if blocking is None else blocking
        ttnn.execute_trace(self.device, self._trace_id, blocking=wait)

    def close(self) -> None:
        """Release the trace. Safe to call more than once."""
        if self._trace_id is None:
            return
        trace_id, self._trace_id = self._trace_id, None
        ttnn.release_trace(self.device, trace_id)

    def __enter__(self) -> "CapturedTrace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def capture(device, record: Callable[[], R]) -> tuple[CapturedTrace, R]:
    """Record `record()`'s device work into a trace and return it with the
    value `record` produced.

    `record` must dispatch only device work: no host read, and nothing whose
    program is not already in the program cache (capture records commands, so
    a first-time kernel compile inside it is not recorded). Warm the cache by
    running the same work eagerly once beforehand.

    Nothing has executed when this returns. The recorded output buffers still
    hold whatever was in them at allocation time, so the caller replays once
    before reading them.
    """
    trace_id = ttnn.begin_trace_capture(device)
    try:
        result = record()
        ttnn.end_trace_capture(device, trace_id)
    except BaseException as e:
        # BaseException, not Exception: a KeyboardInterrupt inside the
        # recording region wedges the chip exactly as hard as a device error.
        _abort_capture(device, trace_id)
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        if HOST_READ_DURING_CAPTURE in str(e):
            raise TraceCaptureError(
                "trace capture failed because the captured work reads a tensor back "
                "to host (.numpy(), to_host(...), .cpu(), .item()), which ttnn does "
                "not support during capture. Move every host read after the replay, "
                "or run this step eagerly. Device error: " + str(e)
            ) from e
        raise TraceCaptureError(f"trace capture failed: {e}") from e
    return CapturedTrace(device, trace_id), result
