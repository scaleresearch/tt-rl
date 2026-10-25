"""The train layer on a real chip: a run, a resume, teardown and eval."""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pytest

from tests._fakes.chips import candidate_chips
from tests._fakes.train_layer import TOY_OBS, state_arrays, toy_train_config
from ttrl.buffers.sampling import PriorityConfig
from ttrl.engine import acquire
from ttrl.spaces import DEFAULT_LEAF, ObsGroup
from ttrl.train.checkpoint import L0, load_checkpoint
from ttrl.train.restore import agent_from_checkpoint
from ttrl.train.run import run
from ttrl.train.trainer import Trainer

#: The one chip this module trains on, as a `DeviceConfig.chips` list. Derived
#: from the visible scope rather than named: on a four-chip box the suite
#: scopes itself to whichever chip is free, and a hardcoded 0 passes only when
#: that happens to be chip 0. These are physical ids, which is what
#: `DeviceConfig.chips` takes; `acquire` is given no id at all and takes the
#: first free chip in the visible set, which under a one-chip scope is it.
OWN_CHIP = candidate_chips(1)

pytestmark = [
    pytest.mark.tt,
    pytest.mark.skipif(not OWN_CHIP, reason="needs one visible Tenstorrent chip"),
]


def _config(tmp_path, name, **kwargs):
    return toy_train_config(tmp_path / name, **kwargs)


def test_the_device_is_released_on_the_exception_path(tmp_path):
    with pytest.raises(ZeroDivisionError, match="forced"):
        with Trainer(_config(tmp_path, "raised", chips=OWN_CHIP), None, None):
            raise ZeroDivisionError("forced")
    # Released, or this second run could not open the same chip.
    assert run(_config(tmp_path, "after", total_frames=16, chips=OWN_CHIP))["frame_count"] == 16


def test_resume_continues_the_same_learner(tmp_path):
    """A real DQN resume, which means the checkpoint carries the transitions."""
    first = run(_config(tmp_path, "a", total_frames=32, checkpoint_replay=True))
    path = str(tmp_path / "a" / "checkpoint.npz")
    saved = load_checkpoint(path)
    assert saved["frame"] == first["frame_count"] == 32
    assert saved["obs_spec"] == ObsGroup.of(TOY_OBS)
    before = state_arrays(saved["agent_state"])
    assert any(k.startswith("optimizer.m") for k in before)

    cfg = _config(tmp_path, "b", total_frames=64, checkpoint_replay=True)
    with acquire(cfg=cfg.device.engine_config()) as (device, profile):
        with Trainer(cfg, device, profile, resume_path=path) as trainer:
            assert trainer.frame_count == 32
            buffer = trainer.replicas.lead.agent.buffer
            assert buffer.size == saved["replay_state"]["pos"].sum(), (
                "the resumed learner must hold the transitions the file carries, "
                "not refill them by warm-up behind an annealed epsilon")
            restored = state_arrays(trainer.replicas.lead.agent.state_dict())
            assert set(restored) == set(before)
            for key, value in before.items():
                assert np.array_equal(value, restored[key]), key
            second = trainer.run()
    assert second["frame_count"] == 64


def test_a_teardown_failure_does_not_mask_the_real_failure(tmp_path):
    class Wedged:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            raise OSError("agent close failed")

    cfg = _config(tmp_path, "teardown")
    with acquire(cfg=cfg.device.engine_config()) as (device, profile):
        trainer = Trainer(cfg, device, profile)
        trainer.replicas.replicas[0].agent = Wedged(trainer.replicas.replicas[0].agent)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ZeroDivisionError, match="the real failure"):
                with trainer:
                    raise ZeroDivisionError("the real failure")
        messages = [str(w.message) for w in caught]
    assert any("agent close failed" in m for m in messages)
    assert any("already propagating" in m for m in messages)


def test_evaluate_scores_the_trained_agent(tmp_path):
    cfg = _config(tmp_path, "eval", total_frames=16)
    with acquire(cfg=cfg.device.engine_config()) as (device, profile):
        with Trainer(cfg, device, profile) as trainer:
            trainer.run()
            result = trainer.evaluate()
    assert result["n_episodes"] == 2
    assert len(result["scores"]) == 2
    assert result["min"] <= result["mean"] <= result["max"]


def test_a_checkpoint_rebuilds_a_working_agent(tmp_path):
    """The device half of the restore claim."""
    config = toy_train_config(tmp_path / "trained", total_frames=32)
    result = run(config)
    assert result["frame_count"] == 32
    assert result["replicas"] == 1 and result["updates"] > 0
    assert result["profile"]["frames_per_s"] > 0
    assert (tmp_path / "trained" / "metrics.jsonl").read_text().strip()
    path = str(tmp_path / "trained" / "checkpoint.npz")
    saved = load_checkpoint(path)

    with acquire(cfg=config.device.engine_config()) as (device, profile):
        agent, restored, meta = agent_from_checkpoint(path, device, profile=profile)
        try:
            weights = agent.state_dict()["online"]
            for key, value in weights.items():
                assert np.array_equal(
                    np.asarray(value),
                    np.asarray(saved["agent_state"]["online"][key]),
                ), key
            actions = agent.act(
                np.zeros((2, *agent.obs_spec[DEFAULT_LEAF].shape), np.float32),
                deterministic=True,
            )
        finally:
            agent.close()
    assert meta["frame"] == 32
    assert actions.shape == (2,)
    assert agent.action_spec.n == 3
    assert restored.algo_cfg == config.algo_cfg


def test_checkpoint_tiers_differ_in_what_they_carry_and_what_they_allow(tmp_path):
    """L1 resumes and L0 only scores. One run feeds every claim."""

    config = toy_train_config(tmp_path / "tiers", total_frames=32)
    with acquire(cfg=config.device.engine_config()) as (device, profile):
        with Trainer(config, device, profile) as trainer:
            trainer.run()
            l1_path = trainer.save_checkpoint(str(tmp_path / "l1.npz"))
            l0_path = trainer.save_checkpoint(str(tmp_path / "l0.npz"), level=L0)
        resumable, snapshot = load_checkpoint(l1_path), load_checkpoint(l0_path)
        assert "optimizer" in resumable["agent_state"]
        assert "optimizer" not in snapshot["agent_state"]
        assert snapshot["level"] == L0 and "frame" not in snapshot

        agent, _config, _meta = agent_from_checkpoint(
            l0_path, device, config=config, profile=profile)
        try:
            assert agent.act(
                np.zeros((2, *agent.obs_spec[DEFAULT_LEAF].shape), np.float32),
                deterministic=True).shape == (2,)
        finally:
            agent.close()
        with pytest.raises(ValueError, match="carries no frame count"):
            Trainer(config, device, profile, resume_path=l0_path)

        agent, _config, _meta = agent_from_checkpoint(
            l1_path, device, config=config, profile=profile, weights_only=True)
        try:
            weights = agent.state_dict(weights_only=True)["online"]
            for key, value in weights.items():
                assert np.array_equal(
                    np.asarray(value),
                    np.asarray(resumable["agent_state"]["online"][key])), key
        finally:
            agent.close()


def _prioritized_losses(tmp_path, mode: str) -> np.ndarray:
    """A whole prioritized DQN run's logged losses, under one step mode.

    `log_interval=1` because the comparison below is over every loss the run
    produced, and a thinned record stream would compare a subsequence.
    """
    config = _config(tmp_path, mode, total_frames=320, log_interval=1)
    config = dataclasses.replace(config, algo_cfg=dataclasses.replace(
        config.algo_cfg, step_mode=mode,
        priority=PriorityConfig(alpha=0.8, beta_start=0.4),
    ))
    run(config)
    lines = (tmp_path / mode / "metrics.jsonl").read_text().splitlines()
    records = (json.loads(line) for line in lines)
    return np.array([r["train/loss"] for r in records if "train/loss" in r])


def test_a_prioritized_run_learns_the_same_thing_traced_as_eager(tmp_path):
    """The run-level guard on per-sample metric readback.

    A prioritized run feeds each sample's TD error back into the priority
    table, so a metric that reads its tile padding instead of its values
    writes zeros there and the run samples a different distribution from the
    next trace replay onward, with every loss still a valid loss and nothing
    raising. Only a whole run shows it: driving one `TrainStep` replays the
    trace too few times, and the loss of any single step is correct either
    way. Measured bit-identical here; with the packed readback restored for
    multi-element metrics the sequences diverge from the third logged loss by
    15.8.
    """
    eager = _prioritized_losses(tmp_path, "eager")
    traced = _prioritized_losses(tmp_path, "traced_fwd_bwd")
    assert len(eager) > 8, "too few updates to replay the trace"
    np.testing.assert_allclose(
        traced, eager, rtol=0, atol=1e-5,
        err_msg="a prioritized run sampled a different distribution traced "
                "than eager, which means a per-sample metric did not reach "
                "the priority table intact",
    )
