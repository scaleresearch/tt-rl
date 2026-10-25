"""What a checkpoint carries of the environment: the observation transform."""

from __future__ import annotations

import json
import os

import numpy as np

from tests._fakes.policy_state import (FakeDevice, config, reward_scale,
                                       train, welford)
from tests._fakes.train_layer import RecordingLogger
from ttrl.train.checkpoint import LATEST, load_checkpoint, read_meta
from ttrl.train.trainer import Trainer


def test_the_running_statistics_survive_a_write_and_reload(tmp_path, agents):
    _cfg, latest, live = train(tmp_path)
    stored = welford(load_checkpoint(latest)["env_policy_state"])
    expected = welford(live)

    assert expected["count"] > 1.0, "the run must have warmed the statistics"
    assert np.allclose(stored["mean"], expected["mean"])
    assert np.allclose(stored["m2"], expected["m2"])
    assert stored["count"] == expected["count"]


def test_the_reward_scale_survives_too_but_only_at_l1(tmp_path, agents):
    """The other running estimator, and the other side of the split."""
    cfg = config(tmp_path, normalize_reward=True)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        live = trainer.replicas.lead.env.save_run_state()
        snapshot = trainer.save_checkpoint(str(tmp_path / "w.npz"), level="L0")
    latest = os.path.join(cfg.log_dir, LATEST)

    stored = reward_scale(load_checkpoint(latest)["env_reward_state"])
    expected = reward_scale(live)
    assert expected["count"] > 1.0, "the run must have warmed the scale"
    assert stored["count"] == expected["count"]
    assert np.allclose(stored["m2"], expected["m2"])
    assert stored["gamma"] == 0.9

    l0 = load_checkpoint(snapshot)
    assert "env_reward_state" not in l0, "a weights snapshot carries no run state"
    assert welford(l0["env_policy_state"])["count"] > 1.0, (
        "the OBSERVATION transform is part of the policy and still travels"
    )


def test_trajectory_state_is_not_in_the_file_at_all(tmp_path, agents):
    """The regression guard for the ALE blob."""

    _cfg, latest, _live = train(tmp_path, name="toy_stacked")
    arrays, meta = read_meta(os.path.realpath(latest))

    snapshot = meta["env_policy_state"]["__list__"]
    assert len(snapshot) == 1, "only the transform-bearing layer travels"
    name, fields = snapshot[0]["__list__"]
    assert name == "NormalizeObservation"
    assert set(fields) == {"mean", "m2", "count", "clip", "epsilon"}

    assert "frames" not in json.dumps(meta["env_policy_state"]), (
        "a frame stack is trajectory state and must not travel"
    )
    assert "__bytes__" not in json.dumps(meta), (
        "no opaque bytes may reach a checkpoint"
    )
    assert not any("frames" in k or "env_policy_state" not in k and
                   k.startswith("env_") for k in arrays)
    assert sorted(k for k in arrays if k.startswith("env_policy_state")) == [
        "env_policy_state.0.1.m2", "env_policy_state.0.1.mean",
    ]

