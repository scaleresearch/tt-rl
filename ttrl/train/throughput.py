"""A run measuring itself: rounds, updates, frames and wall time.

Always on. Timing a round costs two `perf_counter` calls, so there was never a
reason to time only some of them.

Frames are counted from what the loop REPORTS rather than from the round size,
because the last round of a run can be short.
"""

from __future__ import annotations

import json
import time

__all__ = ["Throughput"]


class Throughput:
    """Wall-clock throughput, accumulated over a run's rounds."""

    def __init__(self, frame: int = 0) -> None:
        self._from = frame
        self.rounds = 0
        self.updates = 0
        self.frames = 0
        self.wall_s = 0.0

    def round_ended(self, frame: int, elapsed: float) -> None:
        self.rounds += 1
        self.wall_s += elapsed
        self.frames += frame - self._from
        self._from = frame

    def updated(self) -> None:
        self.updates += 1

    def report(self) -> dict[str, float] | None:
        """The measurement, or None when there is nothing measured yet."""
        if self.rounds == 0 or self.wall_s <= 0.0:
            return None
        return {
            "rounds": self.rounds,
            "updates": self.updates,
            "frames": self.frames,
            "wall_s": self.wall_s,
            "frames_per_s": self.frames / self.wall_s,
            "s_per_round": self.wall_s / self.rounds,
        }


def write_report(path: str, report: dict[str, float] | None, *,
                 frame: int) -> None:
    """The machine-readable throughput report, if there is one and a path for it.

    Here rather than in the trainer because what the file holds is this
    module's measurement; the trainer owns only when it is written, which is
    from its teardown ladder.
    """

    if not path or report is None:
        return
    with open(path, "w") as fh:
        json.dump({**report, "frame_count": frame, "wrote_at": time.time()},
                  fh, indent=2)
