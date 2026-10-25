"""Fakes for the train/ layer: a toy environment and host-only agents.

The environment is a real `ttrl.envs.base.Env` behind the real
`RecordEpisodeStatistics` wrapper and is registered in the real env registry,
so a test drives the production construction path rather than a substitute
for it. The agents implement the `Agent` protocol and own nothing but numpy.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ttrl.algos.clock import UpdateClock
from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.scaling import divide_for_replicas
from ttrl.envs import ENVS, register
from ttrl.envs.wrappers import RecordEpisodeStatistics
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec
from ttrl.train.checkpoint.types import register_checkpoint_type
from ttrl.train.config import DeviceConfig, EnvConfig, TrainConfig

TOY_OBS = ObsSpec.vector(4)
TOY_GROUP = ObsGroup.of(TOY_OBS)
TOY_ACTIONS = ActionSpec.discrete(3)


class ToyEnv:
    """Deterministic 4-vector observations, fixed-length episodes."""

    def __init__(self, *, episode_len: int = 5, reward: float = 1.0) -> None:
        if episode_len < 1:
            raise ValueError("episode_len must be >= 1")
        self.episode_len = episode_len
        self.reward = reward
        self._t = 0
        self._seed = 0
        self.closed = False

    @property
    def episode_bound(self) -> int:
        """`ttrl.envs.bounds`: episodes are exactly this long."""
        return self.episode_len

    @property
    def obs_spec(self) -> ObsSpec:
        return TOY_OBS

    @property
    def action_spec(self) -> ActionSpec:
        return TOY_ACTIONS

    def _obs(self) -> np.ndarray:
        return np.full(4, float(self._seed * 100 + self._t), dtype=np.float32)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        return self._obs(), self.reward, terminated, False, {}

    def close(self) -> None:
        self.closed = True


def make_toy(**kwargs):
    return RecordEpisodeStatistics(ToyEnv(**kwargs))


if "toy" not in ENVS:
    register("toy", make_toy)


@register_checkpoint_type
@dataclasses.dataclass(frozen=True)
class ToyAlgoConfig:
    """The two seams a trainer reads off an algorithm's config, and no more."""

    NETWORK_BATCH_FIELD = "batch_size"
    REPLICA_SCALED_FIELDS = ("buffer_capacity",)

    batch_size: int = 8
    buffer_capacity: int = 64

    @property
    def network_batch(self) -> int:
        return self.batch_size

    def scale_for_replicas(self, n: int) -> "ToyAlgoConfig":
        return divide_for_replicas(self, n, self.REPLICA_SCALED_FIELDS)


def toy_train_config(log_dir, *, total_frames: int = 32, chips=1, n_envs: int = 2,
                     **overrides):
    """A short real DQN run on the toy env. Imports the algorithm's own config.

    Names no chip. `chip=None` lets the run scope itself to whichever chip is
    free, which is the only thing a test can correctly ask for: a scoped
    process sees one chip and it is not always chip 0. `chips`, when it is a
    list, takes PHYSICAL ids and a caller derives those from the visible scope
    (see `tests._fakes.chips.candidate_chips`).
    """

    base = dict(
        algo="dqn",
        total_frames=total_frames,
        eval_interval=10_000,
        eval_episodes=2,
        checkpoint_interval=total_frames,
        log_dir=str(log_dir),
        seed=0,
        env=EnvConfig(name="toy", n_envs=n_envs, kwargs={"episode_len": 6}),
        device=DeviceConfig(chip=None, chips=chips),
        algo_cfg=DQNConfig(
            batch_size=8, buffer_capacity=512, learning_starts=16, train_freq=4,
            n_step=1, target_sync_interval=4, eps_decay_frames=64,
        ),
    )
    return TrainConfig(**{**base, **overrides})


def state_arrays(state, prefix="") -> dict:
    """Every ndarray in a nested agent state, keyed by its dotted path."""
    out = {}
    for key, value in state.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(state_arrays(value, path + "."))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, np.ndarray):
                    out[f"{path}.{i}"] = item
        elif isinstance(value, np.ndarray):
            out[path] = value
    return out


class ScriptedAgent:
    """Host-only Agent. Updates every `update_every` frames, counts everything.

    The cadence is the REAL `ttrl.algos.clock.UpdateClock`, which is why this
    fake can tell `update()` CALLS from the gradient steps each call is worth.
    A hand-rolled stand-in counted one step per call, so a replica starved of
    clock credit looked identical to one doing a full round's work, and that
    is what hid the multi-replica defect. `ttrl.algos.clock` carries no ttnn,
    so importing it keeps this fake host-only.
    """

    def __init__(
        self,
        obs_spec=TOY_GROUP,
        action_spec: ActionSpec = TOY_ACTIONS,
        *,
        update_every: int = 1,
        learning_starts: int = 0,
        loss: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.update_every = update_every
        self.learning_starts = learning_starts
        self._clock = UpdateClock(update_every, learning_starts)
        self.loss = loss
        self.weight = np.zeros(4, dtype=np.float32)
        self.moment = np.zeros(4, dtype=np.float32)
        self.step = 0
        self.observed: list[Any] = []
        #: `update()` calls.
        self.updates = 0
        #: Gradient steps those calls were worth, which is the quantity a
        #: replica can be starved of while its call count looks healthy.
        self.gradient_steps = 0
        self.closes = 0
        self._rng = np.random.default_rng(seed)

    # -- Agent protocol -----------------------------------------------------

    def act(self, obs, *, deterministic: bool = False,
            episode_starts=None) -> np.ndarray:
        n = int(obs.lanes)
        return np.zeros(n, dtype=np.int64) if deterministic else self._rng.integers(
            0, self.action_spec.n, size=n, dtype=np.int64
        )

    def observe(self, transition) -> None:
        self.observed.append(transition)

    def should_update(self, frame: int) -> bool:
        self._clock.advance(frame)
        return self._clock.due

    def update(self) -> dict[str, float]:
        steps = max(1, self._clock.owed)
        self._clock.consume(steps)
        self.updates += 1
        self.gradient_steps += steps
        self.step += 1
        self.weight += 1.0
        self.moment += 0.5
        return {"loss": self.loss, "step": float(self.step),
                "gradient_steps": float(steps)}

    def state_dict(self, *, weights_only: bool = False) -> dict:
        state = {"weight": self.weight.copy(), "nested": {"a": [1, 2]}}
        if weights_only:
            return state
        # The clock rides along, as it does for the real agents: without it a
        # resumed run accrues credit for every frame before the checkpoint.
        return {**state, "step": self.step, "moment": self.moment.copy(),
                "clock": self._clock.state_dict()}

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        # The real agents' asymmetry: a weights-only state handed to a plain
        # load raises rather than restarting the optimizer at zero, while
        # weights_only=True against a full state is how a score reads a
        # resumable file.
        if not weights_only and "step" not in state:
            raise ValueError(
                "ScriptedAgent.load_state_dict: this state carries weights only "
                "and no optimizer state. Continuing would restart the moments "
                "at zero. Pass weights_only=True to score it."
            )
        self.weight = np.asarray(state["weight"], dtype=np.float32).copy()
        if not weights_only:
            self.step = int(state["step"])
            self.moment = np.asarray(state["moment"], dtype=np.float32).copy()
            self._clock.load_state_dict(state["clock"])

    def close(self) -> None:
        self.closes += 1

    # -- ParamSync surface --------------------------------------------------

    def parameters(self):
        return [self.weight]

    def named_parameters(self):
        return [("weight", self.weight)]


class FailingAgent(ScriptedAgent):
    """Raises on the nth update, or on close, to drive the teardown ladder."""

    def __init__(self, *args, fail_on_update: int | None = None,
                 fail_on_close: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_on_update = fail_on_update
        self.fail_on_close = fail_on_close

    def update(self) -> dict[str, float]:
        metrics = super().update()
        if self.fail_on_update is not None and self.updates >= self.fail_on_update:
            raise ZeroDivisionError("the real failure")
        return metrics

    def close(self) -> None:
        super().close()
        if self.fail_on_close:
            raise OSError("teardown failure")


class NanAgent(ScriptedAgent):
    def update(self) -> dict[str, float]:
        super().update()
        return {"loss": float("nan")}


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[dict, int]] = []
        self.flushes = 0
        self.closed = 0

    def log(self, metrics: dict, step: int) -> None:
        self.records.append((dict(metrics), step))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed += 1

    def keys(self) -> set[str]:
        return {k for metrics, _ in self.records for k in metrics}
