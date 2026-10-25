"""The one loop, at one replica and at several."""

from __future__ import annotations

import pytest

from tests._fakes.train_layer import (
    NanAgent,
    RecordingLogger,
    ScriptedAgent,
    make_toy,
)
from ttrl.algos.base import AutoresetTracker
from ttrl.algos.clock import UpdateClock
from ttrl.envs.vector import SyncVectorEnv
from ttrl.train.cadence import Cadence, check_cadence
from ttrl.train.loop import Replica, run_loop


def _replica(index=0, n_envs=1, episode_len=4, **agent_kwargs):
    env = SyncVectorEnv([lambda: make_toy(episode_len=episode_len)] * n_envs)
    return Replica(ScriptedAgent(**agent_kwargs), env, index=index, seed=index)


def _drive(replicas, total_frames, logger=None, frame=0):
    logger = logger or RecordingLogger()
    rounds = []
    reached = run_loop(
        replicas,
        total_frames=total_frames,
        frame=frame,
        on_metrics=logger.log,
        on_round_end=lambda f, elapsed: rounds.append((f, elapsed)),
    )
    return reached, rounds, logger


def test_one_replica_reaches_total_frames():
    r = _replica()
    reached, rounds, logger = _drive([r], 10)
    assert reached == 10
    assert len(rounds) == 10
    assert r.agent.updates == 10
    assert r.agent.gradient_steps == 10
    r.env.close()


@pytest.mark.parametrize("n", [1, 2, 4])
def test_every_replica_does_the_same_gradient_work(n):
    """Lockstep in gradient STEPS, which is the quantity replicas can be     starved of."""
    rs = [_replica(index=i, update_every=1, n_envs=2) for i in range(n)]
    total_frames = 160
    _drive(rs, total_frames)
    steps = [r.agent.gradient_steps for r in rs]
    want = total_frames // 1
    assert steps == [want] * n, f"n={n} starved some replica: {steps}"
    assert sum(steps) == want * n
    for r in rs:
        r.env.close()


def test_a_round_is_worth_more_than_one_gradient_step_here():
    """The discriminating power of the test above, asserted rather than assumed."""
    r = _replica(update_every=1, n_envs=2)
    _drive([r], 160)
    assert r.agent.updates == 80
    assert r.agent.gradient_steps == 160
    assert r.agent.gradient_steps > r.agent.updates
    r.env.close()


def test_on_update_fires_only_on_updating_rounds_and_carries_the_metrics():
    """The hook sees the update it fires after."""
    r = _replica(learning_starts=6)
    fired = []
    logged = []
    run_loop(
        [r],
        total_frames=10,
        frame=0,
        on_metrics=lambda m, s: logged.append(m),
        on_round_end=lambda f, e: None,
        on_update=fired.append,
    )
    train = [m for m in logged if any(k.startswith("train/") for k in m)]
    assert fired == train and all(fired)
    assert len(fired) == 4
    r.env.close()


@pytest.mark.parametrize("n", [1, 2])
def test_the_update_predicate_is_asked_once_per_round_on_every_replica(n):
    rs = [_replica(index=i, n_envs=4) for i in range(n)]
    asked: dict[int, list[int]] = {}
    for r in rs:
        real, seen = r.agent.should_update, asked.setdefault(r.index, [])

        def counting(frame, real=real, seen=seen):
            seen.append(frame)
            return real(frame)

        r.agent.should_update = counting
    _drive(rs, 8 * n)
    assert asked == {i: [4 * n, 8 * n] for i in range(n)}
    assert [r.agent.updates for r in rs] == [2] * n
    for r in rs:
        r.env.close()


def test_check_cadence_refuses_a_cadence_that_accrues_per_ask():
    """The object half of the same invariant, as `ttrl.check_cadence` sees it."""

    class Healthy:
        def __init__(self):
            self.clock = UpdateClock(4, 0)

        def should_update(self, frame):
            self.clock.advance(frame)
            return self.clock.due

        def update(self):
            self.clock.consume(self.clock.owed)
            return {"gradient_steps": 1.0}

    class AccruesPerAsk(Healthy):
        def should_update(self, frame):
            self.clock.advance(self.clock._frame + 4)
            return self.clock.due

    class Batched(Healthy):
        """One step per frame, taken in batches, as DQN.update does."""

        def __init__(self):
            self.clock = UpdateClock(1, 0)

        def update(self):
            steps = max(1, self.clock.owed)
            self.clock.consume(steps)
            return {"gradient_steps": float(steps)}

    class AnswersPerAsk(Healthy):
        def __init__(self):
            super().__init__()
            self.asks = 0

        def should_update(self, frame):
            self.clock.advance(frame)
            self.asks += 1
            return self.clock.due and self.asks % 2 == 0

    check_cadence(Healthy)
    # Four steps a coarse call and two a fine one is equal work, so the work
    # count accepts what counting calls refused.
    check_cadence(Batched)
    with pytest.raises(AssertionError, match="87.5%"):
        check_cadence(AccruesPerAsk)
    with pytest.raises(AssertionError, match="asked 2 times, answered"):
        check_cadence(AnswersPerAsk)


def test_the_agent_sees_every_step_in_order():
    """The loop hands over every step; the agent drops the padding ones."""

    r = _replica(episode_len=3)
    _drive([r], 8)
    dones = [bool(t.done[0]) for t in r.agent.observed]
    assert dones == [False, False, True, False, False, False, True, False]
    tracker = AutoresetTracker(1)
    valids = [bool(tracker.valid(t)[0]) for t in r.agent.observed]
    assert valids == [True, True, True, False, True, True, True, False]
    r.env.close()


def test_nan_metrics_halt_the_run_after_logging():
    env = SyncVectorEnv([lambda: make_toy()])
    r = Replica(NanAgent(), env, index=0, seed=0)
    logger = RecordingLogger()
    with pytest.raises(FloatingPointError, match="non-finite"):
        _drive([r], 10, logger=logger)
    assert any("error" in m for m, _ in logger.records)
    env.close()


def test_cadence_is_anchored_to_the_absolute_frame_grid():
    resumed = Cadence(100, frame=190)
    assert resumed.last == 100
    assert resumed.due(200) is True
    assert resumed.last == 200


def test_cadence_marker_advances_by_whole_intervals():
    c = Cadence(10)
    fires = [f for f in range(3, 123, 3) if c.due(f)]
    assert c.last == 120
    assert len(fires) == 12


def test_the_loop_asks_the_cadence_and_defaults_it_to_the_agent():
    """`Replica.cadence` is the swap point, and its default is the agent."""
    env = SyncVectorEnv([lambda: make_toy(episode_len=4)])
    agent = ScriptedAgent(update_every=1)
    plain = Replica(agent, env, index=0, seed=0)
    assert plain.cadence is agent

    class EveryOtherFrame:
        """A cadence outside the agent, accruing from the frame it is told."""

        def __init__(self, agent) -> None:
            self.agent = agent
            self.seen = 0
            self.owed = 0

        def should_update(self, frame: int) -> bool:
            if frame > self.seen:
                self.owed += frame // 2 - self.seen // 2
                self.seen = frame
            return self.owed > 0

        def update(self) -> dict:
            owed, self.owed = self.owed, 0
            for _ in range(owed):
                self.agent.update()
            return {"gradient_steps": float(owed)}

    cadence = EveryOtherFrame(ScriptedAgent(update_every=1))
    replica = Replica(cadence.agent, env, index=0, seed=0, cadence=cadence)
    _drive([replica], 8)
    assert cadence.seen == 8
    # The agent's own clock was never consulted: every step came from the
    # injected cadence.
    assert cadence.agent.updates == 4
