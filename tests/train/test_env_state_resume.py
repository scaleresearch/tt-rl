"""What a resume does with the policy state a checkpoint carries.

Three questions, and they are not the same question. Does a resume restore the
statistics and score on them? Does it refuse a file that cannot supply state
this env needs? Does it refuse a config whose transform layers are not the
checkpoint's?

THE MIDDLE ONE WAS A VERSION TEST. `env_policy_state is None` means only that
the file was written before that member existed, and it was refused on that
alone. An Atari stack declares `POLICY_STATE = ()` on every layer and so has
nothing to restore, which made the refusal fire on resumes that had provably
nothing to lose: the same env writing the same checkpoint today stores an
empty snapshot and resumes. It cost a real 250,000 frames.
`ttrl.train.restore` now asks what
this run's env carries, and both halves of that are tested below.

What the file carries at all is `tests/train/test_env_state.py`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests._fakes.policy_state import (
    FakeDevice,
    config,
    strip_policy_state,
    reward_scale,
    train,
    welford,
)
from tests._fakes.train_layer import RecordingLogger
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.train import builders
from ttrl.train.checkpoint import LATEST
from ttrl.train.config import EnvConfig
from ttrl.train.trainer import Trainer



# --------------------------------------------------------------------------
# a resume continues with the warm statistics
# --------------------------------------------------------------------------


def test_a_resume_restores_the_statistics(tmp_path, agents):
    """Both running estimators, which are restored by different members.

    The observation transform comes back through `env_policy_state`, because
    the weights are not interpretable without it. The reward scale comes back
    through `env_reward_state`, because a run whose loss changes scale on
    resume is not the run it reports itself as continuing even though the
    weights compute the same function either way.
    """
    cfg = config(tmp_path, normalize_reward=True)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        live = trainer.replicas.lead.env.save_policy_state()
        live_reward = reward_scale(trainer.replicas.lead.env.save_run_state())
    latest = os.path.join(cfg.log_dir, LATEST)

    cfg2 = config(tmp_path / "second", normalize_reward=True)
    with Trainer(
        cfg2, FakeDevice(), object(), resume_path=latest, logger=RecordingLogger()
    ) as trainer:
        resumed = welford(trainer.replicas.lead.env.save_policy_state())
        resumed_reward = reward_scale(
            trainer.replicas.lead.env.save_run_state())
    assert resumed["count"] == welford(live)["count"]
    assert np.allclose(resumed["mean"], welford(live)["mean"])

    assert live_reward["count"] > 1.0, "the run must have warmed the scale"
    assert resumed_reward["count"] == live_reward["count"]
    assert np.allclose(resumed_reward["m2"], live_reward["m2"])


def test_an_eval_after_a_resume_uses_the_warm_statistics(tmp_path, agents,
                                                        monkeypatch):
    """The measurable consequence, read off the eval env's own statistics.

    Scoring does not fit, so what the eval applies is exactly what it was
    given: nothing at all without a resume, and the checkpoint's statistics
    with one. Read from the eval stack rather than from the whitened
    observations, which saturate the clip on this toy either way.
    """
    _cfg, latest, _live = train(tmp_path)

    cold = _eval_stats(config(tmp_path / "cold"), monkeypatch)
    assert cold["count"] == 0.0, "a fresh eval env is the cold baseline"

    warm = _eval_stats(config(tmp_path / "warm"), monkeypatch,
                       resume_path=latest)
    assert warm["count"] == welford(_live)["count"] > 1.0, (
        "an eval after a resume must score on the scale the policy was "
        "trained on, not on statistics restarted at count 0"
    )
    assert np.allclose(warm["mean"], welford(_live)["mean"])


def _eval_stats(cfg, monkeypatch, **kwargs):
    """The eval env's normaliser statistics AFTER one `trainer.evaluate()`.

    A score is taken through a FIXED function: these are the statistics the
    eval was given, unchanged by the episodes it scored. Fitting on them
    would make the transform move between the first episode and the last and
    report that number for a function `ttrl.train.record` and `ttrl.deploy`
    do not apply. See `ttrl.envs.vec_normalize.freeze_observation_transform`.
    """

    built = []
    real_build_env = builders.build_env

    def build_env(c):
        env = real_build_env(c)
        assert isinstance(env, NormalizeObservation), (
            "this toy stack whitens at its top layer; the walk below assumes it")
        built.append(env)
        return env

    monkeypatch.setattr(builders, "build_env", build_env)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger(),
                 **kwargs) as trainer:
        trainer.evaluate()
    # The training env is built first, so the eval env is the last one. Read
    # from the wrapper, which outlives the lanes it closed.
    return built[-1].get_state()


def test_a_resume_from_a_file_with_no_policy_state_is_refused(tmp_path, agents):
    """Refused because THIS env carries statistics the file cannot supply.

    Continuing anyway would restart them at zero and change the function the
    loaded weights compute.
    """
    _cfg, latest, _live = train(tmp_path)
    real = strip_policy_state(latest)

    with pytest.raises(ValueError) as excinfo:
        Trainer(config(tmp_path / "x"), FakeDevice(), object(),
                resume_path=real, logger=RecordingLogger())
    message = str(excinfo.value)
    assert "env_policy_state" in message
    assert "NormalizeObservation" in message, "the message names what carries state"
    assert "evaluate_checkpoint" in message and "afresh" in message


def test_a_resume_from_such_a_file_is_allowed_when_the_env_carries_nothing(
    tmp_path, agents
):
    """The false refusal this replaced, and what it cost.

    `env_policy_state is None` says only that the file predates that member.
    A stack whose every layer declares `POLICY_STATE = ()`, which is every
    Atari stack, has no cross-episode statistics to lose: the same env writing
    the same checkpoint TODAY stores an empty snapshot and resumes, so
    refusing such a file was a version test standing in for a content test.
    It cost a real 250,000 frames of training.
    """
    _cfg, latest, live = train(tmp_path, name="toy")
    assert live == [], "the plain toy stack must carry no policy state"
    real = strip_policy_state(latest)

    with Trainer(config(tmp_path / "x", name="toy"), FakeDevice(), object(),
                 resume_path=real, logger=RecordingLogger()) as trainer:
        assert trainer.frame_count == 12
        assert trainer.replicas.lead.env.save_policy_state() == []


# --------------------------------------------------------------------------
# the mismatches, both directions
# --------------------------------------------------------------------------


def test_resuming_a_normalised_run_under_a_config_without_it_is_refused(
    tmp_path, agents
):
    _cfg, latest, _live = train(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        Trainer(config(tmp_path / "x", name="toy"), FakeDevice(), object(),
                resume_path=latest, logger=RecordingLogger())
    message = str(excinfo.value)
    assert "NormalizeObservation" in message, "the message names the layer"
    assert "config_from_checkpoint" in message, "and what the user can do"


def test_resuming_an_unnormalised_run_under_a_config_with_it_is_refused(
    tmp_path, agents
):
    _cfg, latest, _live = train(tmp_path, name="toy")
    with pytest.raises(ValueError) as excinfo:
        Trainer(config(tmp_path / "x"), FakeDevice(), object(),
                resume_path=latest, logger=RecordingLogger())
    assert "NormalizeObservation" in str(excinfo.value)


def test_a_separately_declared_eval_env_is_warmed_too(tmp_path, agents):
    """The narrowing this used to need is gone. A policy snapshot names only
    transform-bearing layers, so it loads into an eval stack that differs in
    a frame stack or a time limit, and an Atari stack carries none at all.
    Restricting the warm start to `eval_env is None` left a declared eval
    protocol scoring a normalised policy on cold statistics.
    """
    cfg = config(tmp_path, eval_env=EnvConfig(name="toy_norm",
                                              kwargs={"episode_len": 5},
                                              normalize_obs=True))
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        trainer.replicas.lead.agent.seen.clear()
        assert trainer.evaluate()["n_episodes"] == 1
        first = trainer.replicas.lead.agent.seen[0]
    assert not np.allclose(first, 0.0), (
        "a declared eval protocol must still be scored on the trained scale"
    )


def test_such_a_file_is_refused_when_the_env_config_changed(tmp_path, agents):
    """The one case the narrowing must not let through.

    A file carrying no snapshot does not say what the TRAINING stack carried.
    A modern file does, which is why a non-empty snapshot loaded into a
    bearer-free stack is refused. With no snapshot the stored env config is
    the only other evidence, so an env that differs from the checkpoint's is
    refused rather than resumed on an unknown scale.
    """
    _cfg, latest, _live = train(tmp_path, name="toy")
    real = strip_policy_state(latest)

    # Bearer-free on both sides, so the bearer check cannot be what refuses
    # it: the only difference is the env config itself.
    changed = config(tmp_path / "x", name="toy")
    changed.env.kwargs = {"episode_len": 7}
    with pytest.raises(ValueError) as excinfo:
        Trainer(changed, FakeDevice(), object(), resume_path=real,
                logger=RecordingLogger())
    message = str(excinfo.value)
    assert "different environment config" in message
    assert "config_from_checkpoint" in message


def test_such_a_file_resumes_when_only_the_lane_count_or_seed_changed(
    tmp_path, agents
):
    """A seed and a lane count cannot add or remove a transform-bearing layer,
    so comparing whole `EnvConfig`s would refuse for a difference that has no
    bearing on the question."""
    _cfg, latest, _live = train(tmp_path, name="toy")
    real = strip_policy_state(latest)

    reseeded = config(tmp_path / "x", name="toy", seed=7)
    with Trainer(reseeded, FakeDevice(), object(), resume_path=real,
                 logger=RecordingLogger()) as trainer:
        assert trainer.frame_count == 12
