"""The environment contract.

Protocols rather than base classes, so an environment written outside this
repo satisfies the contract without importing it. `ttrl.envs.check` turns
this contract into an executable test for people writing an adapter, and
`check_env` and `check_vec_env` are reachable from `ttrl.envs` itself.

Autoreset convention: NEXT_STEP. When a lane terminates, `step` returns that
lane's final observation together with `terminated` or `truncated` set, and
the *following* `step` returns the first observation of the new episode while
ignoring the action passed for that lane. This matches Gymnasium 1.x vector
environments. Getting it wrong corrupts the transition that a replay buffer
stores, so `check_vec_env` asserts it directly.

Final observations: on any step where a lane reports `terminated` or
`truncated`, `info` must carry `FINAL_OBS_KEY` ("final_obs"), a sequence of
length `num_envs` holding that lane's last observation of the finished
episode and None for every lane that did not end. Under NEXT_STEP the entry
is the same array the step already returned for that lane, so an
implementation gets it for free. It is required because it is the one thing
that distinguishes NEXT_STEP from SAME_STEP autoreset from the outside: a
SAME_STEP implementation overwrites the returned observation with the new
episode's first frame, and `check_vec_env` catches that by comparing the two.
A replay buffer should store the `final_obs` entry as the transition's
next_obs, which is correct under either convention.

Privileged state: an environment may put simulator state in its observation
that the actor must not read, which is what an asymmetric actor-critic trains
against. It declares the division on its `ObsSpec` as fields
(`ObsSpec.split_vector(ObsField("sensed", n), ObsField("sim_state", m,
actor=False))`) and emits the whole observation from `step` and `reset`. Every
reader takes its own half from the spec: the critic the whole observation, the
actor `ObsSpec.actor`. `ttrl.envs.wrappers.PrivilegedObservation` appends such
a field to any vector-observation environment.

Legal actions: an environment where only some actions are legal declares the
mask's width on its `ObsSpec` (`ObsSpec.vector(dim, action_mask=n)`) and
writes the {0, 1} mask into the final n entries of every observation,
terminal observations included. Such an environment SHOULD raise on an
illegal action rather than coerce one: a coerced action is stored as the
action that was taken, which is a transition the environment never made, and
nothing downstream can detect it. The checkers sample only legal actions, so
raising is a contract they hold you to rather than a way to fail.

Episode statistics: a `VecEnv` must carry, on any step where a lane's episode
ends, `EPISODE_KEY` ("episode") holding `{"r": undiscounted return, "l":
length}` in that lane's entry of `LANE_KEY` ("lane"), a list of length
`num_envs`. `ttrl.train.evaluate` reads the score from
`info["lane"][i]["episode"]["r"]` and has nothing else to read it from.

NO WRAPPER IS NEEDED FOR THIS. `SyncVectorEnv` and `AsyncVectorEnv`
accumulate each lane's return and length themselves and emit the key, because
they already see every lane's reward and done flags on every step
(`ttrl.envs.episodes`). A single `Env` therefore supplies nothing: it returns
rewards and done flags and the vector layer does the rest. Only a `VecEnv`
written outside this library has to emit the key itself, and `check_vec_env`
asserts it.

`ttrl.envs.wrappers.RecordEpisodeStatistics` remains for the one case the
vector layer cannot serve: a lane that transforms its own reward must have
the return taken BELOW that transform, and only a wrapper inside the lane
sits there. A lane that supplies the key keeps its own number; the
vector-level accumulator never overwrites it.

Rendering is an optional capability, described in `ttrl.envs.render`. It is
not a member of `Env`, because most environments cannot render; a caller asks
`ttrl.envs.supports_render(env)` and then calls `env.render()`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

__all__ = [
    "Env",
    "VecEnv",
    "FINAL_OBS_KEY",
    "EPISODE_KEY",
    "LANE_KEY",
]

#: `info` key carrying per-lane final observations. See the module docstring.
FINAL_OBS_KEY = "final_obs"

#: `info` key carrying a finished episode's return and length.
EPISODE_KEY = "episode"

#: `VecEnv` `info` key carrying one info dict per lane.
LANE_KEY = "lane"


@runtime_checkable
class Env(Protocol):
    """A single environment instance."""

    @property
    def obs_spec(self) -> ObsSpec | ObsGroup:
        """What `reset` and `step` return: one array, or one per leaf.

        An `ObsSpec` is the single-array case and the observation is that
        array. An `ObsGroup` of named `ObsLeaf`s declares several, which
        differ in shape, dtype or kind, and the observation is then a
        `dict[str, np.ndarray]` with one entry per leaf name. Both are public
        (`ttrl.ObsSpec`, `ttrl.ObsGroup`, `ttrl.ObsLeaf`), the vector layer
        promotes a bare spec to a one-leaf group, and every reader above it
        holds a group, so the two declarations cost the same.
        """

    @property
    def action_spec(self) -> ActionSpec: ...

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        """Start a new episode: its first observation and an info dict.

        `seed` reseeds this environment's own generator and is passed once,
        at the start of a run; a later `reset` with no seed continues the
        sequence. The info dict may be empty, and nothing in the library
        requires a key from `reset`.
        """

    def step(
        self, action: Any
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool,
               dict[str, Any]]:
        """Apply one action: `(obs, reward, terminated, truncated, info)`.

        `terminated` is True when the episode reached an end state of its own
        (a hand played out, a robot fell over), and the state's value is then
        zero, which is what stops a learner bootstrapping through it.
        `truncated` is True when the episode was cut off with a future it
        still had (a time limit, a step budget), and the final state's value
        is bootstrapped as usual. Return exactly one of them, in this order:
        exchanging the two is a correct-looking environment that teaches a
        learner the value of a terminal state, so `ttrl.check_env` warns on
        the pattern.

        `reward` is this step's reward, finite, undiscounted and untransformed.
        `obs` is this step's observation, matching `obs_spec`, and on a
        terminating step it is the episode's FINAL observation: the vector
        layer resets on the following step (NEXT_STEP autoreset, in the module
        docstring) and a single `Env` never resets itself.

        `info` is free for anything the caller logs. One key is reserved:
        `EPISODE_KEY` ("episode"), which a lane supplies only to take its own
        return below a reward transform. The vector layer accumulates the
        return and length itself, so an ordinary `Env` supplies nothing, and
        it is the vector layer that fills `LANE_KEY` ("lane") and
        `FINAL_OBS_KEY` ("final_obs").
        """

    def close(self) -> None:
        """Release what this environment holds. Called once, and may be
        called after an exception, so it tolerates a half-built instance."""


@runtime_checkable
class VecEnv(Protocol):
    """A batch of environments stepped together.

    Observations are stacked on a leading lane axis, so shape is
    `(num_envs, *obs_spec.shape)`. Rewards, terminations and truncations are
    arrays of length `num_envs`. On any step where a lane ends, `info` carries
    `FINAL_OBS_KEY` as described in the module docstring.
    """

    @property
    def num_envs(self) -> int: ...

    @property
    def obs_spec(self) -> ObsSpec | ObsGroup: ...

    @property
    def action_spec(self) -> ActionSpec: ...

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray,
               np.ndarray, dict[str, Any]]: ...

    def close(self) -> None: ...
