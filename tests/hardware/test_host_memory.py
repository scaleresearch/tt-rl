"""The gate on the README's host memory figure."""

from __future__ import annotations

import pytest

from tests._fakes.train_layer import toy_train_config
from tests._memory import assert_no_growth, growth_bytes_per_step, host_rss_bytes
from ttrl.train.run import run

pytestmark = pytest.mark.tt

#: Env steps of the measured run. Long enough that the trend fit has something
#: to fit and the one-off allocations are far behind it, short enough to stay a
#: test.
FRAMES = 3_000

#: Env steps thrown away before sampling starts. The first rounds allocate the
#: things a run allocates once: the program cache filling, the replay buffer's
#: arrays, the trace region, the JSONL sink's buffer.
WARMUP_FRAMES = 600

#: The README's figure, 4.4 MB over 100,000 env steps.
README_BYTES = 4.4e6
README_FRAMES = 100_000
CLAIMED_BYTES_PER_FRAME = README_BYTES / README_FRAMES

#: Headroom over the claimed rate. A short run cannot resolve 44 bytes per step
#: against page-granular RSS, so the ceiling is the claimed rate times this.
#: A real per-step leak is orders of magnitude above the claim, never a factor
#: of four: the leak this guards against is a retained tape or a growing list,
#: which costs kilobytes per step.
HEADROOM = 4.0


class _RssLogger:
    """A real logger whose side effect is a sample. The trainer calls `log`
    once per round under `log_interval=1`, which is the only per-step hook a
    run offers; the run below states that interval so the series is one
    sample per round."""

    def __init__(self):
        self.samples: list[int] = []
        self.frames: list[int] = []

    def log(self, metrics: dict, frame: int) -> None:
        self.frames.append(int(frame))
        self.samples.append(host_rss_bytes())

    def close(self) -> None:
        pass


def test_host_memory_growth_stays_under_the_rate_the_readme_claims(tmp_path):
    logger = _RssLogger()
    config = toy_train_config(tmp_path / "rss", total_frames=FRAMES, n_envs=2,
                              log_interval=1)
    result = run(config, logger=logger)
    assert result["frame_count"] >= FRAMES

    sampled = [(frame, rss) for frame, rss in zip(logger.frames, logger.samples)
               if frame >= WARMUP_FRAMES]
    assert len(sampled) >= 20, (
        f"only {len(sampled)} samples past frame {WARMUP_FRAMES}; the trend fit "
        f"needs a series, and this run logged {len(logger.frames)} times in all"
    )

    series = [rss for _frame, rss in sampled]
    span = sampled[-1][0] - sampled[0][0]
    per_sample = growth_bytes_per_step(series)
    per_frame = per_sample * (len(series) - 1) / span

    ceiling = CLAIMED_BYTES_PER_FRAME * HEADROOM
    assert per_frame <= ceiling, (
        f"host memory grows at {per_frame:.1f} B/env step, which is "
        f"{per_frame * README_FRAMES / 1e6:.2f} MB over the {README_FRAMES:,}-step "
        f"run the README reports {README_BYTES / 1e6} MB for. The ceiling here is "
        f"{ceiling:.1f} B/step. Either a per-step allocation is being retained, "
        f"or the README's figure is no longer true and the table has to change."
    )
    # The level check as well as the trend: a step change partway through a
    # long flat run dilutes below any slope threshold.
    assert_no_growth(series, max_bytes_per_step=ceiling * span / (len(series) - 1),
                     tolerance_bytes=int(ceiling * span), label="host RSS")
