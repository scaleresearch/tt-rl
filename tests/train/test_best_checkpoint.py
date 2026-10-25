"""Best-checkpoint selection: the score, the file, and surviving a resume."""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.config import TrainConfig
from ttrl.train.checkpoint import (
    BEST,
    BestScore,
    L1,
    best_path,
    history_path,
    prune,
    save_checkpoint,
)


def _best_file(log_dir, *, extra, frame=100):
    """A best checkpoint at the name and level the Trainer writes one at."""
    save_checkpoint(
        best_path(str(log_dir)),
        agent_state={"weight": np.zeros(4, dtype=np.float32), "step": 1},
        obs_spec=ObsSpec.vector(4),
        action_spec=ActionSpec.discrete(3),
        level=L1,
        frame=frame,
        config=TrainConfig(total_frames=1000, log_dir=str(log_dir)),
        env_policy_state=[],
        extra=extra,
    )


def _stamp(score, frame, *, metric="eval/mean", mode="max"):
    return {"best_metric": metric, "best_mode": mode,
            "best_score": score, "best_frame": frame}


def test_the_first_eval_always_takes_the_file(tmp_path):
    best = BestScore(str(tmp_path), metric="eval/mean", mode="max")
    assert best.observe({"eval/mean": -500.0}, 100) is True
    assert best.score == -500.0 and best.frame == 100


def test_a_worse_score_does_not_take_the_file(tmp_path):
    best = BestScore(str(tmp_path), metric="eval/mean", mode="max")
    best.observe({"eval/mean": 10.0}, 100)
    assert best.observe({"eval/mean": 9.0}, 200) is False
    assert best.score == 10.0 and best.frame == 100
    assert best.last == 9.0


def test_an_unknown_metric_is_refused_with_the_available_keys(tmp_path):
    best = BestScore(str(tmp_path), metric="eval/success", mode="max")
    with pytest.raises(ValueError, match="eval/median"):
        best.observe({"eval/mean": 1.0, "eval/median": 1.0}, 10)


def test_evals_since_best_counts_only_the_non_improvements(tmp_path):
    best = BestScore(str(tmp_path), metric="eval/mean", mode="max")
    for score in (1.0, 0.5, 0.4, 2.0, 1.9):
        best.observe({"eval/mean": score}, 10)
    assert best.evals == 5
    assert best.evals_since_best == 1


def test_the_best_file_is_not_a_frame_keyed_one_and_survives_pruning(tmp_path):
    """The whole point: the retention bound must not reach the best file."""
    for frame in (10, 20, 30):
        save_checkpoint(
            history_path(str(tmp_path), frame),
            agent_state={"weight": np.zeros(2, dtype=np.float32)},
            obs_spec=ObsSpec.vector(2), action_spec=ActionSpec.discrete(2),
            level="L0", env_policy_state=[],
        )
    (tmp_path / BEST).write_bytes(b"not a checkpoint, just a name")
    removed = prune(str(tmp_path), 1)
    assert len(removed) == 2
    assert (tmp_path / BEST).is_file()


def test_a_worse_score_after_a_resume_does_not_displace_the_best(tmp_path):
    """The failure this exists for: the first eval after a resume is not the
    best the run has seen, only the best this invocation has seen."""
    _best_file(tmp_path, extra=_stamp(42.0, 300), frame=300)
    best = BestScore.resumed(str(tmp_path), metric="eval/mean", mode="max")
    assert best.observe({"eval/mean": 7.0}, 400) is False
    assert best.score == 42.0 and best.frame == 300
    assert best.observe({"eval/mean": 43.0}, 500) is True


def test_a_resume_on_a_different_metric_is_refused(tmp_path):
    _best_file(tmp_path, extra=_stamp(42.0, 300, metric="eval/median"))
    with pytest.raises(ValueError, match="not comparable"):
        BestScore.resumed(str(tmp_path), metric="eval/mean", mode="max")


def test_a_file_at_that_name_carrying_no_score_is_refused(tmp_path):
    """A user's own snapshot saved to the name is not a high-water mark."""
    _best_file(tmp_path, extra={"replicas": 1})
    with pytest.raises(ValueError, match="carries no 'best_score'"):
        BestScore.resumed(str(tmp_path), metric="eval/mean", mode="max")


def test_the_patience_count_restarts_at_a_resume(tmp_path):
    """Nothing in the file says how many evals the earlier run had wasted."""
    _best_file(tmp_path, extra=_stamp(42.0, 300))
    best = BestScore.resumed(str(tmp_path), metric="eval/mean", mode="max")
    assert best.evals_since_best == 0 and best.evals == 0
