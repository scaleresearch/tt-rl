"""Where a run's metrics go.

`Logger` is a protocol the caller satisfied, and the library's default is a
file sink. The library never decides to print: `StdoutLogger` writes to a
stream the CALLER hands it, which is a sink doing its job, and nothing
constructs one unless asked. What it exists for is that a run with no sign of
life is unusable however well it works inside; the run that validated this
framework took 26 minutes and the only progress its author saw came from a
JSONL reader they wrote themselves.

A sink RAISES. A missing `tensorboard` or `wandb` raises and names the extra
that provides it; a failed write raises. A metrics file is a run's only
evidence, and a sink that degrades to silence turns a broken run into a quiet
one.

The two sinks that ship a run's scalars to a VIEWER, and so need a dependency
this library does not install, live in `ttrl.train.trackers` and are exported
from here. The split is along what a sink costs to construct: everything in
this module needs nothing beyond numpy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ttrl.train.trackers import TensorBoardLogger, WandbLogger

__all__ = [
    "EVAL_PREFIX",
    "UPDATE_PREFIX",
    "build_logger",
    "JsonlLogger",
    "Logger",
    "MultiLogger",
    "NullLogger",
    "StdoutLogger",
    "TensorBoardLogger",
    "WandbLogger",
]


#: The key prefix a run's eval scores carry. `ttrl.train.trainer` builds its
#: eval keys from this one name, and `StdoutLogger` reads the same one to
#: decide what it must never drop, so the prefix is stated once.
EVAL_PREFIX = "eval/"

#: The key prefix one gradient step's metrics carry. `ttrl.train.loop` builds
#: its update keys from this one name and `ttrl.train.cadence.UpdateRecords`
#: reads the same one, so what counts as an update record is stated once.
UPDATE_PREFIX = "train/"


@runtime_checkable
class Logger(Protocol):
    """Somewhere a run's scalars go. `step` is the run's frame count."""

    def log(self, metrics: dict[str, Any], step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class NullLogger:
    """Keeps nothing, for a caller that wants a run to write no file.

    The default is `JsonlLogger` on the config's `log_dir`, so this is the
    way to opt out of that rather than the way into it.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(
        f"object of type {type(obj)!r} is not JSON serializable, and a metrics "
        f"record is JSON. Log numbers, strings, bools, None, lists, dicts and "
        f"numpy arrays; convert anything else at the call site."
    )


class JsonlLogger:
    """One JSON object per `log` call, appended to `<log_dir>/metrics.jsonl`.

    Flushed on every record. A buffered tail is lost exactly when it is worth
    most: a run that diverges or halts on a NaN at hour six is the one whose
    last few hundred records explain what happened, and a mid-run read of a
    buffered file is a coin flip that depends on how many records the
    algorithm happened to emit. Measured on this filesystem at 0.79 us per
    record against 0.17 us buffered, so the flush costs 0.6 us, which is
    microseconds against a training round measured in milliseconds.

    `flush` rather than `fsync`: flushing puts the record in the page cache,
    which survives the process dying, and that is the failure this exists
    for. `fsync` measures at 289 us per record on the same filesystem, three
    orders of magnitude more, and buys only machine power loss.

    A write failure raises. It means a full disk or a dead filesystem, and
    both mean the checkpoint write coming up will fail too.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "metrics.jsonl"
        self._fh = open(self.path, "a", buffering=1)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, "wall_time": time.time(), **metrics}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class StdoutLogger:
    """A one-line progress report on a stream the caller supplies.

    Writes `frame 12800/100000 (12.8%) 414 f/s | train/loss 0.0231`. The rate
    is measured over the whole run so far, which is the number that predicts
    when it finishes rather than the one that jitters per round.

    `total_frames` is optional only because a scoring run has no total. Pass
    `config.total_frames` for a training run and the line carries a
    percentage.

    `unit` names what `step` counts, because the `Logger` protocol's `step` is
    a frame count for a training run and an episode count for a scoring run,
    and a progress line labelled in the wrong unit misleads for as long as it
    is believed.

    `min_interval_s` throttles by wall clock, not by frames, because frames
    per second varies by three orders of magnitude between a toy env and a
    traced Atari run and a fixed frame stride floods one and starves the
    other. `UNTHROTTLED` says what the throttle may never drop.
    """

    #: Keys whose record is always written. The throttle exists for the
    #: per-round rollout and train lines, which a round produces whatever the
    #: frame rate; everything here is produced on a cadence the CALLER chose
    #: and so is already as rare as they asked for. An eval score was being
    #: dropped by this throttle for the whole of a run: it lands between two
    #: rollout lines, so almost every one fell inside the interval and stdout
    #: showed a run with no sign of whether it was learning, while the same
    #: record reached a JsonlLogger beside it. `error` is here because the
    #: last thing a halting run says must not be the thing that got dropped,
    #: and `run/` carries the status and the stop reason, which a run says
    #: once.
    UNTHROTTLED = ("error", EVAL_PREFIX, "run/")

    def __init__(
        self,
        total_frames: int | None = None,
        *,
        stream=None,
        min_interval_s: float = 5.0,
        unit: str = "frame",
    ) -> None:
        # The first argument is a FRAME COUNT and the stream is keyword-only,
        # so `StdoutLogger(sys.stdout)` used to construct and then divide a
        # step by a file object about a thousand frames later.
        if total_frames is not None and not isinstance(total_frames, int):
            raise TypeError(
                f"StdoutLogger: the first argument is total_frames, a frame "
                f"count, and got {type(total_frames).__name__} "
                f"({total_frames!r}). The stream is keyword-only: write "
                f"StdoutLogger(cfg.total_frames, stream=sys.stdout), or take "
                f"the JSONL file and the stdout line together with "
                f"ttrl.build_logger(log_dir, total_frames)."
            )
        # Resolved here and not at module scope, so a caller can redirect
        # sys.stdout before constructing one and have it take effect.
        self.stream = sys.stdout if stream is None else stream
        self.total_frames = total_frames
        self.min_interval_s = min_interval_s
        self.unit = unit
        self._started = time.monotonic()
        self._last = 0.0

    def log(self, metrics: dict[str, Any], step: int) -> None:
        now = time.monotonic()
        if (self._last and not self._always(metrics)
                and now - self._last < self.min_interval_s):
            return
        self._last = now
        elapsed = max(now - self._started, 1e-9)
        progress = f"{self.unit} {step}"
        if self.total_frames:
            progress += f"/{self.total_frames} ({100.0 * step / self.total_frames:.1f}%)"
        parts = [progress, f"{step / elapsed:.0f} {self.unit}/s"]
        body = " ".join(f"{k} {_short(v)}" for k, v in sorted(metrics.items()))
        line = " | ".join(parts) + (f" | {body}" if body else "")
        # Written, not printed: the stream is the caller's and `print` would
        # reach whatever sys.stdout happened to be at the time.
        self.stream.write(line + "\n")
        self.flush()

    def _always(self, metrics: dict[str, Any]) -> bool:
        """Whether this record is one the throttle may not drop."""
        return any(key.startswith(self.UNTHROTTLED) for key in metrics)

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        # The stream belongs to the caller, so it is flushed and not closed.
        self.flush()


def _short(value: Any) -> str:
    if isinstance(value, (bool, int, str)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return repr(value)


class MultiLogger:
    """Fan one `log` call out to several sinks, in order.

    `close` closes every sink even if one raises, and re-raises the first
    failure, so one broken sink never strands the others' file handles.
    """

    def __init__(self, *sinks: Logger) -> None:
        self.sinks = list(sinks)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for sink in self.sinks:
            sink.log(metrics, step)

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        first: BaseException | None = None
        for sink in self.sinks:
            try:
                sink.close()
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first


def build_logger(
    log_dir: str | Path,
    total_frames: int | None = None,
    *extra: Logger,
    stream=None,
    min_interval_s: float = 5.0,
    unit: str = "frame",
) -> MultiLogger:
    """The two sinks a watched run wants: the JSONL file, and a stdout line.

    `build_logger(cfg.log_dir, cfg.total_frames)` is the whole of it, and a
    third sink is one more argument:
    `build_logger(cfg.log_dir, cfg.total_frames, WandbLogger("proj"))`. It
    exists because both shipped examples wrote their own JSONL-plus-print
    class rather than compose these three names, which is what an awkward
    surface looks like from outside.

    Takes the two numbers rather than a `TrainConfig`, so this module stays
    clear of the config schema. `extra` sinks are logged after these two.
    """
    return MultiLogger(
        JsonlLogger(log_dir),
        StdoutLogger(total_frames, stream=stream,
                     min_interval_s=min_interval_s, unit=unit),
        *extra,
    )
