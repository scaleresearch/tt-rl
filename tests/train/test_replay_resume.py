"""What a resume does with the transitions a checkpoint carries, or does not."""

from __future__ import annotations

import dataclasses
import inspect
import os

import numpy as np
import pytest

from tests._fakes.policy_state import FakeDevice
from tests._fakes.train_layer import TOY_ACTIONS, TOY_GROUP, RecordingLogger
from ttrl.algos import ALGOS
from ttrl.algos.offpolicy import ReplayWriter
from ttrl.buffers import ReplayBuffer
from ttrl.buffers.persist import (
    load_replay_state,
    owns_replay,
    replay_state as snapshot,
    replay_state,
)
from ttrl.buffers.sampling import PriorityConfig
from ttrl.train import builders
from ttrl.train.checkpoint import L0, LATEST, load_checkpoint, save_checkpoint
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer

LANES = 2


class ReplayAgent:
    """A host-only agent that owns a real `ReplayBuffer`, like DQN and SAC."""

    def __init__(self, obs_spec=TOY_GROUP, action_spec=TOY_ACTIONS, *,
                 priority=None) -> None:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.buffer = ReplayBuffer(obs_spec, action_spec, capacity=64,
                                   n_envs=LANES, n_step=1, seed=0,
                                   priority=priority)
        self._writer = ReplayWriter(self.buffer, LANES)
        self.weight = np.zeros(4, dtype=np.float32)
        self.closed = False

    def act(self, obs, *, deterministic=False, episode_starts=None):
        return np.zeros(int(obs.lanes), dtype=np.int64)

    def observe(self, transition) -> None:
        self._writer.observe(transition)

    def should_update(self, frame: int) -> bool:
        return False

    def update(self) -> dict:
        return {}

    def state_dict(self, *, weights_only: bool = False) -> dict:
        return {"weight": self.weight.copy()}

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        self.weight = np.asarray(state["weight"], np.float32).copy()

    def replay_state(self) -> dict:
        return replay_state(self.buffer)

    def load_replay_state(self, state: dict) -> None:
        load_replay_state(self.buffer, state)

    def close(self) -> None:
        self.closed = True

    def parameters(self):
        return [self.weight]

    def named_parameters(self):
        return [("weight", self.weight)]


@pytest.fixture
def replay_agents(monkeypatch):
    """Install a `build_agent` whose agents own a replay buffer.

    `install.built` collects them, which is how a test reaches an agent built
    by a construction that then raised.
    """
    def install(priority=None):
        def build(cfg, *, obs_spec, action_spec, device, profile=None,
                  algo_cfg=None):
            agent = ReplayAgent(obs_spec, action_spec, priority=priority)
            install.built.append(agent)
            return agent

        monkeypatch.setattr(builders, "build_agent", build)

    install.built = []
    return install


def config(log_dir, *, replay: bool = False, total_frames: int = 24) -> TrainConfig:
    return TrainConfig(
        total_frames=total_frames,
        eval_interval=10_000,
        eval_episodes=1,
        checkpoint_interval=total_frames,
        checkpoint_replay=replay,
        log_dir=str(log_dir),
        env=EnvConfig(name="toy", n_envs=LANES, kwargs={"episode_len": 3}),
    )


def _run(tmp_path, *, replay: bool, total_frames: int = 24):
    cfg = config(tmp_path / "run", replay=replay, total_frames=total_frames)
    with Trainer(cfg, FakeDevice(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
        return cfg, os.path.join(cfg.log_dir, LATEST), trainer.replicas.lead.agent


def test_a_resume_without_the_transitions_is_refused_by_name(tmp_path,
                                                             replay_agents):
    """The whole point of gap 1."""
    owning = {name for name in ALGOS.names() if owns_replay(ALGOS.resolve(name))}
    assert owning, "no shipped algorithm owns a replay buffer; the refusal is dead"
    offering = {name for name in ALGOS.names()
                if _declares_priority(ALGOS.resolve(name))}
    assert offering == owning, (
        f"prioritized replay is offered by {sorted(offering)} and a replay "
        f"buffer is owned by {sorted(owning)}; PER belongs to exactly the "
        f"algorithms that have a replay buffer to prioritize"
    )

    replay_agents()
    _cfg, latest, live = _run(tmp_path, replay=False)
    assert live.buffer.size > 0, "the run must have collected something to lose"
    assert load_checkpoint(latest)["replay_state"] is None

    with pytest.raises(ValueError) as caught:
        Trainer(config(tmp_path / "x"), FakeDevice(), object(),
                resume_path=latest, logger=RecordingLogger())
    message = str(caught.value)
    assert "replay_state" in message, "the message names the missing member"
    assert "checkpoint_replay" in message, "and the setting that would store it"
    assert "EMPTY buffer" in message and "annealed" in message
    assert "evaluate_checkpoint" in message and "afresh" in message
    # The refusal is intentional; leaving the run it already opened behind is
    # not. The agent holds a captured trace on the chip and the env may hold
    # worker processes, so a refused construction that closes neither makes
    # the chip look held to every later process. `tests/engine/
    # test_cleanup_paths.py` gates the unwind stack itself.
    refused = replay_agents.built[-1]
    assert refused is not live, "the resume built its own agent"
    assert refused.closed, "the refused construction left its agent open"


def _declares_priority(builder) -> bool:
    """Whether this algorithm's own config has a `priority` field."""
    parameter = inspect.signature(builder).parameters.get("cfg")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return False
    return any(f.name == "priority"
               for f in dataclasses.fields(type(parameter.default)))


@pytest.mark.parametrize("priority", [None, PriorityConfig(alpha=0.7)])
def test_a_resume_restores_every_member_of_the_buffer(tmp_path, replay_agents,
                                                      priority):
    """Every member `replay_state` writes, ENUMERATED from the live object."""
    replay_agents(priority)
    _cfg, latest, live = _run(tmp_path, replay=True)
    before = live.replay_state()
    size = live.buffer.size
    assert size > 0
    col = int(live.buffer._pos[0])
    live.buffer.add(np.zeros(4, np.float32), 0, 12345.0, True, False,
                    final_obs=np.ones(4, np.float32), lane=0)
    assert before["rewards"][0, col] != 12345.0, "the snapshot aliases the ring"
    assert not bool(before["ends"][0, col])
    assert (0, col) not in set(zip(before["boundaries"]["lanes"].tolist(),
                                   before["boundaries"]["cols"].tolist()))

    with Trainer(config(tmp_path / "second", replay=True), FakeDevice(), object(),
                 resume_path=latest, logger=RecordingLogger()) as trainer:
        warm = trainer.replicas.lead.agent
        after = warm.replay_state()
        assert warm.buffer.size == size
        drawn = warm.buffer.sample(
            8, beta=None if priority is None else 0.5).obs["obs"]

    with pytest.raises(ValueError, match="weights snapshot"):
        save_checkpoint(str(tmp_path / "l0.npz"), agent_state={}, obs_spec=TOY_GROUP,
                        action_spec=TOY_ACTIONS, level=L0, replay_state=before)
    narrower = ReplayBuffer(TOY_GROUP, TOY_ACTIONS, capacity=32, n_envs=LANES,
                            n_step=1, seed=0, priority=priority)
    with pytest.raises(ValueError, match="per_lane_capacity"):
        load_replay_state(narrower, before)

    assert set(after) == set(before), "a member of the buffer was not restored"
    assert ("priorities" in before) == (priority is not None)
    # `open_successors` is the one member a resume CONSUMES rather than
    # restores: closing the seam spends it. This run ends on an episode
    # boundary so it is empty on both sides, and the mid-episode case is the
    # test below.
    assert before["open_successors"] == {} == after["open_successors"]
    _assert_same(before, after, "replay_state")
    assert drawn.shape[0] == 8


def test_a_mid_episode_resume_closes_the_seam_instead_of_splicing(
    tmp_path, replay_agents
):
    """The transition at the seam must not be paired with a reset observation.

    A lane's successor is the next column of its own ring, so the last
    transition stored mid-episode took its successor from whatever was
    written next. `run_loop` resets every environment on resume, and no
    simulator state is in the file, so what is written next is the FIRST
    observation of a new episode. Measured on a 2-lane toy DQN run
    checkpointed 3 frames into an episode: the stored transition's successor
    came back as the post-reset observation (the toy env's step counter 0)
    instead of the observation it really led to (counter 1), one transition
    per lane per resume, and an n-step window would have summed rewards
    across the join.

    Closed as a truncation against the recorded successor, so the transition
    keeps the exact value target it had and the window stops there.
    """
    replay_agents()
    # 5 rounds of 2 lanes at episode_len 3: each lane's last write is the
    # first step of its second episode, which is mid-episode.
    _cfg, latest, live = _run(tmp_path, replay=True, total_frames=10)
    seam = {lane: int(live.buffer._pos[lane]) - 1 for lane in range(LANES)}
    stored = load_checkpoint(latest)["replay_state"]
    assert set(stored["open_successors"]) == {"0", "1"}, (
        "both lanes were mid-episode, so the file must say what followed")
    for lane, col in seam.items():
        assert not bool(stored["ends"][lane, col]), "the seam was left open"

    with Trainer(config(tmp_path / "second", replay=True), FakeDevice(), object(),
                 resume_path=latest, logger=RecordingLogger()) as trainer:
        warm = trainer.replicas.lead.agent
        for lane, col in seam.items():
            assert bool(warm.buffer.ends[lane, col]), (
                f"lane {lane}'s open episode was not closed at the seam")
            assert not bool(warm.buffer.dones[lane, col]), (
                "a truncation, not an invented termination")
            successor = warm.buffer._final_obs.read(lane, col)["obs"]
            follows = np.asarray(
                stored["open_successors"][str(lane)]["obs"], np.float32)
            assert np.array_equal(successor, follows), (
                "the closed boundary must carry the observation the step "
                "really led to")
        trainer.replicas.lead.reset()
        trainer.replicas.lead.step()
        for lane, col in seam.items():
            nxt = (col + 1) % warm.buffer.per_lane_capacity
            after_reset = np.asarray(warm.buffer.obs["obs"][lane, nxt])
            assert not np.array_equal(
                after_reset, warm.buffer._final_obs.read(lane, col)["obs"]), (
                "the next column holds a new episode's first observation, "
                "which is exactly why the seam column may not read it")

    # The other half of the same guarantee: a snapshot of a buffer filled
    # past an episode boundary by a route that records no successor is
    # refused where it is taken, since no reader could close it.

    bypassed = ReplayBuffer(TOY_GROUP, TOY_ACTIONS, capacity=64, n_envs=LANES,
                            n_step=1, seed=0)
    bypassed.add_batch(TOY_GROUP.observation(np.zeros((LANES, 4), np.float32)),
                       np.zeros(LANES, np.int64), np.zeros(LANES, np.float32),
                       np.zeros(LANES, bool))
    with pytest.raises(ValueError, match="note_open_successor"):
        snapshot(bypassed)

    without = {k: v for k, v in stored.items() if k != "open_successors"}
    with pytest.raises(ValueError, match="open_successors") as caught:
        load_replay_state(live.buffer, without)
    assert "afresh" in str(caught.value), "the refusal names a way out"


def _assert_same(want, got, path: str) -> None:
    """Compare two trees of the same shape, naming the member that differs."""
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path} differs in its keys"
        for key in want:
            _assert_same(want[key], got[key], f"{path}.{key}")
        return
    if isinstance(want, np.ndarray):
        assert np.array_equal(got, want), f"{path} came back different"
        return
    assert got == want, f"{path} came back as {got!r}, not {want!r}"
