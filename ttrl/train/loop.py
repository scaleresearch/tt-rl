"""The training loop. One loop, parameterised by the number of replicas.

There is one loop because there is one thing to do: step every replica's
environment, hand each agent what its step produced, let each replica's
cadence say when it wants to learn, and average the replicas' parameters when
they do. n=1 is that with the replica loop running once and no averaging.

Nothing here knows an algorithm. Off-policy and on-policy differ in what
`Agent.observe` keeps and what `Agent.should_update` answers, which is the
agent's business; a buffer, a replay ratio, a rollout length and an epoch
count never appear below. The autoreset padding step is likewise the agent's
to drop (`ttrl.algos.base.AutoresetTracker`), derived from the flags it
already sees rather than from a field this loop adds.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from ttrl.algos.base import Transition
from ttrl.train.logging import UPDATE_PREFIX

__all__ = ["Replica", "StepSource", "run_loop"]


#: The key prefix an update's metrics carry. Stated once here, where the keys
#: are built, because a per-update record is the one a run emits per gradient
#: step and `ttrl.train.trainer` reads the same name to put those records on
#: `TrainConfig.log_interval`, resolved as `log_interval_frames`.



@runtime_checkable
class StepSource(Protocol):
    """Where a round's transitions come from. A vector env is one of these.

    Declared because it was already a live duck type: `Replica` below calls
    exactly the first five members and nothing else in this module touches an
    environment, so anything answering them trains, whether it steps
    gymnasium, replays a logged dataset or drives a real robot. `num_envs` is
    how many frames one `step` advances the run by, which is the only thing
    the loop reads it for.

    THE REST IS WHAT A RUN, RATHER THAN THE LOOP, ASKS OF ITS SOURCE, and it
    is declared here because a source is injected into a run
    (`ttrl.train.trainer.Trainer(source=...)`) and a run checkpoints, scores
    and releases whatever it is stepping. `close()` is the teardown ladder's
    (`ttrl.train.teardown`). The two state pairs are what a checkpoint carries
    and a resume restores: `save_policy_state`/`load_policy_state` is the
    transform state the weights are not interpretable without, and
    `save_run_state`/`load_run_state` is the run statistics that are no part
    of what the weights compute (`ttrl.train.checkpoint.file` says what each
    means). A source carrying neither answers `[]` to both saves and ignores
    both loads, which is five short methods and an ANSWER rather than an
    absence: the alternative, a run that skips the transfer for a source that
    does not declare it, is the silently-worse resume `ttrl.train.resume`
    exists to refuse.

    WHAT `reset` AND `step` RETURN, which a source written from scratch cannot
    read off these signatures. `reset(seed=...)` gives
    `(observation, info)` and `step(actions)` gives
    `(observation, rewards, terminated, truncated, info)`, with `rewards`,
    `terminated` and `truncated` arrays of length `num_envs`.

    `observation` is a `ttrl.Observation` rather than a bare array, because
    every reader above the loop takes one: build it through the spec, which is
    the only thing that validates it, as
    `obs_spec.observation({DEFAULT_LEAF: rows})` for a single-leaf source
    (`ttrl.DEFAULT_LEAF`) and one array per leaf name otherwise. Each array's
    leading axis is the lane axis, of length `num_envs`.

    `info` must carry `"lane"` (`ttrl.envs.LANE_KEY`), a list of `num_envs`
    per-lane info dicts, and a lane that just ended an episode must carry
    `{"r": return, "l": length}` under `"episode"`
    (`ttrl.envs.EPISODE_KEY`) in its own entry. That is where every episode
    metric is read from and the only place: `ttrl.train.loop.episode_metrics`
    and `ttrl.train.evaluate` both look there, so a source that omits the key
    trains normally and reports no return for the whole run.
    `ttrl.envs.SyncVectorEnv` accumulates both for its lanes; a source of your
    own supplies them itself.

    A step source that is a real environment also has an env contract to keep
    (`ttrl.check_env`): the two are separate because this is what a run
    needs and that is what an observation has to be.
    """

    obs_spec: Any
    action_spec: Any
    num_envs: int

    def reset(self, *, seed: int | None = None) -> tuple: ...

    def step(self, actions) -> tuple: ...

    def close(self) -> None: ...

    def save_policy_state(self) -> list: ...

    def load_policy_state(self, snapshot: list) -> None: ...

    def save_run_state(self) -> list: ...

    def load_run_state(self, snapshot: list) -> None: ...


class Replica:
    """One agent, its environment, and the carried state of their stepping.

    `cadence` is the `ttrl.train.cadence.UpdateCadence` that decides when this
    replica learns, and defaults to the agent, which is one already. One
    cadence per replica and never a shared instance: a cadence accrues credit
    from the frames ITS replica was handed, and the cross-replica starvation
    `check_cadence` measures is exactly what a shared one reintroduces.
    """

    def __init__(self, agent, env: StepSource, *, index: int,
                 seed: int | None, cadence=None) -> None:
        self.agent = agent
        self.env = env
        self.index = index
        self.seed = seed
        self.cadence = agent if cadence is None else cadence
        self.obs = None

    def reset(self) -> None:
        self.obs, _info = self.env.reset(seed=self.seed)

    def step(self) -> tuple[Transition, dict]:
        actions = self.agent.act(self.obs)
        result = self.env.step(actions)
        transition = Transition.from_step(self.env.obs_spec, self.obs,
                                          actions, result)
        self.agent.observe(transition)
        self.obs = result[0]
        return transition, result[4]


def episode_metrics(info: dict) -> dict[str, float]:
    """Episode returns and lengths from the lanes that finished this step.

    `RecordEpisodeStatistics` puts `{"r", "l"}` under `info["episode"]` on a
    lane's terminal step, and the vector env collects per-lane infos under
    `info["lane"]`.
    """
    ended = [
        lane["episode"]
        for lane in info.get("lane", ())
        if isinstance(lane, dict) and "episode" in lane
    ]
    if not ended:
        return {}
    return {
        "rollout/episode_return": float(np.mean([e["r"] for e in ended])),
        "rollout/episode_length": float(np.mean([e["l"] for e in ended])),
        "rollout/episodes": float(len(ended)),
    }


def run_loop(
    replicas: list[Replica],
    *,
    total_frames: int,
    frame: int,
    on_metrics: Callable[[dict, int], None],
    on_round_end: Callable[[int, float], str | None],
    on_update: Callable[[dict], None] | None = None,
) -> int:
    """Train until `frame` reaches `total_frames`; return the frame reached.

    `on_update(metrics)` is called once per round in which the replicas
    updated, and is where parameter averaging happens. `metrics` is the lead
    replica's own update metrics under the same `train/` keys the logger
    receives, so a hook reading them and a metrics file read afterwards talk
    about the same numbers. `on_round_end(frame, elapsed)` gets the
    round's wall time and runs the frame-keyed periodic work (eval,
    checkpoint), which is INSIDE the loop's view rather than outside it: both
    run inline and block, so a hang in either stops the run just as dead as a
    hang in an update.

    `on_round_end` ENDS THE RUN by returning a reason. The loop logs it and
    breaks, so an early stop leaves by the same path as a run that reached its
    budget, with every later step (the final checkpoint, the teardown ladder)
    still taken. Returning None continues.

    Every replica is asked whether to update, at the SAME frame, and the round
    updates only if they all say yes. One shared frame is what keeps them in
    lockstep: asking each about its own frame count would let replica 0 cross
    a threshold up to n-1 increments before the last replica and start the
    replicas learning on different frames, an asymmetry the averaging would
    then smear together. Asking only replica 0 is the other failure and the
    one this used to have; see the comment on the call.
    """
    for replica in replicas:
        replica.reset()

    while frame < total_frames:
        t0 = time.perf_counter()
        for replica in replicas:
            _transition, info = replica.step()
            frame += replica.env.num_envs
            metrics = episode_metrics(info)
            if metrics and replica.index == 0:
                on_metrics(metrics, frame)

        # EVERY replica is asked, not replica 0 on everyone's behalf. Asking
        # is how a replica learns the frame it has reached, and an off-policy
        # cadence sizes `update()` from that; a replica never asked holds no
        # credit and takes one gradient step per round however much data the
        # round carried. On two chips the arithmetic costs replica 1 87.5%
        # of its work.
        decisions = [
            replica.cadence.should_update(frame) for replica in replicas
        ]
        # All or none, so averaging never mixes replicas that learned a
        # different number of times. `all` rather than replica 0's answer: a
        # replica says no only while its buffer holds too little to sample,
        # and credit already accrued is kept rather than consumed, so the
        # round is deferred and nothing is lost.
        updated = all(decisions)
        lead_metrics: dict = {}
        if updated:
            for replica in replicas:
                metrics = replica.cadence.update()
                if replica.index == 0:
                    lead_metrics = {
                        f"{UPDATE_PREFIX}{k}": v for k, v in metrics.items()
                    }
                    on_metrics(lead_metrics, frame)
                _require_finite(metrics, replica, on_metrics, frame)

        if updated and on_update is not None:
            on_update(lead_metrics)

        reason = on_round_end(frame, time.perf_counter() - t0)
        if reason:
            # Logged here rather than by the caller, so the metrics file
            # records why the run ended whoever supplied the condition.
            on_metrics({"run/stop_reason": reason}, frame)
            break
    return frame


def _require_finite(metrics: dict, replica: Replica, on_metrics, frame: int) -> None:
    """Halt on a non-finite metric.

    A NaN loss poisons every subsequent update, so the run stops at the first
    one rather than burning hours producing garbage.
    """
    bad = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not np.isfinite(v)
    }
    if not bad:
        return
    # Logged before raising, so the metrics file records why the run ended.
    on_metrics({"error": f"non-finite metrics on replica {replica.index}"}, frame)
    raise FloatingPointError(
        f"{type(replica.cadence).__name__}.update() produced non-finite metrics on "
        f"replica {replica.index}: {bad}. The run is halted because a non-finite "
        "loss poisons every later update. Lower the learning rate, check the "
        "reward scale, or clip the gradient norm."
    )
