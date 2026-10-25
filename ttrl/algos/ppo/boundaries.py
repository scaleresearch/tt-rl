"""Where a recurrent PPO's episode-start flags come from, as one public seam.

`ttrl.algos.ppo.recurrent` says that during training the flags are PPO's own
business: `observe` sees every step's `terminated` and `truncated`, so the
flag for the next step is the previous step's boundary and no caller supplies
it. That is the right DEFAULT and it was the only behaviour there was, which
left the first question anyone with a recurrent policy asks -- is the policy
actually using its memory? -- answerable only by assigning to
`RecurrentRuntime._boundary` and `RolloutBuffer._carried_boundary`. Both are
private, and the second belongs to `ttrl.buffers`, which is not part of the
`ttrl` surface at all.

So the flags get a declared source. `EpisodeBoundaries` is consulted once per
`observe` and its answer fans out to BOTH consumers: the acting carry's reset
mask for the next step, and the `starts` column the sequence unroll masks
with during the update. One hook, because one wrong answer in two places is
the drift that has no symptom: a run whose acting resets and whose learning
resets disagree trains a policy nobody evaluates.

THE MEMORY-DEFEATED ABLATION ARM is `boundaries` returning ones: every step
begins an episode, so the hidden state is masked away before every cell, in
acting and in the unroll alike. The arm that measured -18.85 on single-frame
Pong against -2.30 recurrent and -13.65 feedforward is that class, five
lines, through this seam.

WHAT A HOOK MAY NOT DO, which is why its answer is checked rather than
trusted. It is host-side numpy, called between device steps, so it may read
the host freely and must return a plain lane-major array: the flags are
copied into the carry's bound `start` buffer and into the buffer's carried
row IN PLACE, and a hook handing back a device tensor, a wider array than the
lane count fixed at trace capture, or a non-finite value is refused by name.
Nothing here ever runs inside a captured step.

`check_boundaries` is the contract test to run against your own. It checks
the mechanics and judges no policy: returning `transition.terminated` alone
is a legal hook and a classic silent bug (it leaks state across every
time-limit truncation), and returning ones is a legal hook and a deliberate
ablation. The mechanics are this repo's business; which flags your experiment
wants are yours.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

import numpy as np

from ttrl.algos.base import Transition
from ttrl.spaces import DEFAULT_LEAF, ObsGroup, ObsSpec

__all__ = ["EpisodeBoundaries", "TransitionBoundaries", "check_boundaries"]


@runtime_checkable
class EpisodeBoundaries(Protocol):
    """Which lanes just ended an episode, and so where memory resets.

    Hand one to `PPO(boundaries=...)`, or assign it to `PPO.boundaries`
    between rollouts. A memoryless PPO has nothing to reset and refuses one
    at construction rather than accepting a hook it would never call; an
    assignment that is not one is named at the next `observe`.
    """

    def boundaries(self, transition: Transition) -> np.ndarray:
        """The boundary flags for `transition`, lane-major, `(n_envs,)`.

        1.0 where this step ended that lane's episode, which makes the NEXT
        step a fresh one. Values between 0.0 and 1.0 scale the state that
        survives, which is what a leaky-memory ablation wants; outside that
        range there is no masking to do and the answer is refused.

        Called once per stored step, including autoreset padding steps, and
        before the step is recorded. `transition` is read-only: the arrays in
        it are the run's own and a hook that writes through them changes what
        every other reader of that step sees.
        """
        ...


class TransitionBoundaries:
    """The default: a boundary is `terminated or truncated`.

    `terminated` alone is the wrong answer and `transition.done` is why this
    class is three lines: a time-limit truncation ends an episode for the
    memory even though it still bootstraps, so the step after one belongs to a
    different episode and must not see the previous one's state.
    """

    def boundaries(self, transition: Transition) -> np.ndarray:
        return np.asarray(transition.done, dtype=np.float32).reshape(-1)

    def __repr__(self) -> str:
        return "TransitionBoundaries()"


def require_boundaries(boundaries, *, recurrent: bool, who: str):
    """The members and the applicability, checked where one is accepted."""
    if boundaries is None:
        return TransitionBoundaries()
    if not callable(getattr(boundaries, "boundaries", None)):
        raise TypeError(
            f"{who}: {type(boundaries).__name__} is not an EpisodeBoundaries: "
            f"it has no callable 'boundaries'. Give it "
            f"boundaries(transition) -> flags, lane-major and 1.0 where that "
            f"lane's episode just ended (ttrl.check_boundaries runs one "
            f"against the contract)."
        )
    if not recurrent:
        raise ValueError(
            f"{who}: this policy has no memory, so there are no episode-start "
            f"flags for {type(boundaries).__name__} to supply and nothing "
            f"would ever call it. Set PPOConfig.recurrent, or drop the "
            f"boundaries argument."
        )
    return boundaries


def read_boundaries(boundaries, transition: Transition, *,
                    n_envs: int) -> np.ndarray:
    """The hook's answer for one step, as flags PPO can use, or raise.

    Returns a fresh contiguous float32 array, so what the carry and the
    rollout buffer copy in place is never the caller's own buffer: a hook
    reusing one scratch row would otherwise rewrite a boundary already
    recorded, and `starts` would describe a step that never happened.
    """
    who = type(boundaries).__name__
    hook = getattr(boundaries, "boundaries", None)
    if not callable(hook):
        raise TypeError(
            f"PPO.boundaries holds {who}, which is not an EpisodeBoundaries: "
            f"it has no callable 'boundaries'. Assign one with "
            f"boundaries(transition) -> flags, lane-major and 1.0 where that "
            f"lane's episode just ended (ttrl.check_boundaries runs one "
            f"against the contract)."
        )
    answer = hook(transition)
    if answer is None:
        raise TypeError(
            f"{who}.boundaries returned None for a transition of {n_envs} "
            f"lanes. Return the flags: lane-major, 1.0 where that lane's "
            f"episode just ended, e.g. "
            f"np.asarray(transition.done, np.float32)."
        )
    if _is_device_tensor(answer):
        raise TypeError(
            f"{who}.boundaries returned a device tensor. The flags are host "
            f"numpy: they are written into the acting carry's bound `start` "
            f"buffer in place, at the address a captured trace recorded, so a "
            f"tensor here would either rebind that buffer or need a host read "
            f"inside the step. Return numpy, e.g. "
            f"np.asarray(transition.done, np.float32)."
        )
    try:
        flags = np.asarray(answer, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{who}.boundaries returned {type(answer).__name__}, which is not "
            f"an array of float flags ({exc}). Return "
            f"np.asarray(..., np.float32) of one flag per lane."
        ) from None
    if flags.shape != (n_envs,):
        raise ValueError(
            f"{who}.boundaries returned {flags.shape} flags and this run has "
            f"{n_envs} lanes. The lane count is fixed when the acting step is "
            f"captured, so a different width cannot be masked at all. Return "
            f"one flag per lane, shaped ({n_envs},)."
        )
    if not np.isfinite(flags).all():
        raise ValueError(
            f"{who}.boundaries returned a non-finite flag "
            f"({flags.tolist()}). The flag multiplies the hidden state "
            f"entering the step, so a NaN or an inf erases the memory of "
            f"every later step in the sequence and the loss reduces to NaN. "
            f"Return finite flags in [0.0, 1.0]."
        )
    if flags.min() < 0.0 or flags.max() > 1.0:
        raise ValueError(
            f"{who}.boundaries returned a flag outside [0.0, 1.0] "
            f"(min {float(flags.min())}, max {float(flags.max())}). 1.0 "
            f"resets the lane's memory, 0.0 keeps it and a value between "
            f"scales what survives; there is nothing outside that range for "
            f"the mask to mean. Clip the answer, or return booleans cast to "
            f"float32."
        )
    return np.ascontiguousarray(flags, dtype=np.float32)


def _is_device_tensor(answer) -> bool:
    module = str(type(answer).__module__)
    return module.startswith("ttnn") or module.startswith("ttrl.engine")


def check_boundaries(boundaries, *, n_envs: int = 4) -> None:
    """Hold an `EpisodeBoundaries` to its contract, or raise saying how.

    Drives the hook over the four shapes of step a run actually produces:
    nothing ending, every lane ending, a termination in one lane only, and a
    truncation in one lane only. Every answer goes through the same
    `read_boundaries` a real `observe` uses, so the refusals are the run's
    own and not a second copy of the rules.

    Safe to call on the instance a run will use: it reads the hook and the
    transitions it builds here, and touches no agent.
    """
    require_boundaries(boundaries, recurrent=True, who="check_boundaries")
    who = type(boundaries).__name__
    before = set(sys.modules)
    for terminated, truncated in _CASES(n_envs):
        transition = _transition(terminated, truncated)
        signature = _fingerprint(transition)
        read_boundaries(boundaries, transition, n_envs=n_envs)
        if _fingerprint(transition) != signature:
            raise AssertionError(
                f"{who}.boundaries wrote through the transition it was "
                f"handed: terminated or truncated changed under it. Those "
                f"arrays are the step the run is about to record and every "
                f"other reader sees the edit, so the stored `dones` would "
                f"disagree with what the environment reported. Copy before "
                f"you modify: np.asarray(transition.done, np.float32)."
            )
    new = [name for name in set(sys.modules) - before
           if name == "ttnn" or name.startswith("ttnn.")]
    if new:
        raise AssertionError(
            f"{who}.boundaries imported ttnn ({new}), and the flags are "
            f"host-side numpy between device steps. Nothing here may touch a "
            f"device: the answer is copied into buffers a captured trace "
            f"already recorded the addresses of. Keep the hook in numpy."
        )


def _CASES(n_envs: int):
    """The four step shapes, as `(terminated, truncated)` per lane."""
    zeros = np.zeros(n_envs, dtype=bool)
    one = np.zeros(n_envs, dtype=bool)
    one[0] = True
    last = np.zeros(n_envs, dtype=bool)
    last[-1] = True
    return ((zeros, zeros), (np.ones(n_envs, dtype=bool), zeros),
            (one, zeros), (zeros, last))


def _transition(terminated: np.ndarray, truncated: np.ndarray) -> Transition:
    n_envs = int(terminated.shape[0])
    spec = ObsGroup.of(ObsSpec.vector(2))
    obs = spec.observation({DEFAULT_LEAF: np.zeros((n_envs, 2), np.float32)})
    return Transition(
        obs=obs, action=np.zeros(n_envs, np.int64),
        reward=np.zeros(n_envs, np.float32),
        terminated=terminated.copy(), truncated=truncated.copy(),
        next_obs=obs,
    )


def _fingerprint(transition: Transition) -> bytes:
    return (np.asarray(transition.terminated).tobytes()
            + np.asarray(transition.truncated).tobytes())
