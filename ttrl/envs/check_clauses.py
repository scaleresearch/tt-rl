"""One clause of the environment contract each, for the checkers to drive.

Split from `ttrl.envs.check`, which is the DRIVING: a reset, a step loop and
the order they happen in. Everything here is a single assertion about one
return value, taking the thing it judges and the `where` it names in the
message, so both checkers hold a single environment and a vector one to the
same clause and neither carries its own idea of what the contract says.

What every clause asserts is stated in `ttrl.envs.base`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ttrl.envs.base import EPISODE_KEY, FINAL_OBS_KEY, LANE_KEY
from ttrl.spaces import ActionSpec, ObsGroup, draw_legal, group_mask_rows

__all__ = ["check_episode", "check_final_obs", "check_lane_episodes",
           "check_mask", "check_obs", "sample_action", "warn_if_swapped"]


def check_obs(obs, group: ObsGroup, *, lanes: int | None, where: str):
    """`obs` as an `Observation`, or a refusal naming what the spec requires."""
    try:
        observation = group.observation(obs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"{where} returned an observation this obs_spec refuses: {exc}"
        ) from exc
    if observation.lanes != lanes:
        raise ValueError(
            f"{where} returned {observation.lanes} lanes, obs_spec requires "
            f"{lanes}"
        )
    return observation


def check_mask(obs, group: ObsGroup, *, where: str) -> np.ndarray:
    """The legal-action rows of `obs`, refusing a state with no legal action."""
    rows = group_mask_rows(obs, group)
    empty = np.flatnonzero(~(rows > 0.5).any(axis=-1))
    if empty.size:
        raise ValueError(
            f"{where} row {int(empty[0])} declares every action illegal, so "
            f"there is nothing for an agent to choose. End the episode instead "
            f"of presenting a state with no legal move."
        )
    return rows


def sample_action(
    spec: ActionSpec,
    rng: np.random.Generator,
    lanes: int | None,
    obs_spec,
    obs,
):
    if spec.kind == "discrete":
        if obs_spec.action_mask:
            # Legal actions only. Raising on an illegal action is the
            # documented recommendation, so a checker that fed uniform actions
            # would fail every honest implementation of a masked environment.
            picks = draw_legal(rng, check_mask(obs, obs_spec, where="action mask"))
            return picks[0] if lanes is None else picks
        size = () if lanes is None else (lanes,)
        return rng.integers(0, spec.n, size=size, dtype=np.int64)
    if spec.kind == "multi_discrete":
        sizes = np.asarray(spec.nvec, dtype=np.int64)
        shape = sizes.shape if lanes is None else (lanes, sizes.shape[0])
        return rng.integers(0, np.broadcast_to(sizes, shape), dtype=np.int64)
    shape = (spec.dim,) if lanes is None else (lanes, spec.dim)
    return rng.uniform(spec.low_array, spec.high_array, size=shape).astype(np.float32)


def check_final_obs(
    info: dict[str, Any],
    obs,
    done: np.ndarray,
    group: ObsGroup,
    *,
    where: str,
) -> None:
    """Assert the final-observation contract for the lanes that just ended."""
    ended = np.flatnonzero(done).tolist()
    if not isinstance(info, dict) or FINAL_OBS_KEY not in info:
        raise ValueError(
            f"{where} ended lanes {ended} but info carries no {FINAL_OBS_KEY!r}. "
            f"Every step that reports a lane done must supply that lane's final "
            f"observation under {FINAL_OBS_KEY!r}, because it is what lets a "
            f"caller tell NEXT_STEP autoreset from SAME_STEP."
        )
    finals = info[FINAL_OBS_KEY]
    if len(finals) != done.shape[0]:
        raise ValueError(
            f"{where} info[{FINAL_OBS_KEY!r}] has length {len(finals)}, "
            f"expected one entry per lane ({done.shape[0]})"
        )
    for lane in ended:
        final = finals[lane]
        if final is None:
            raise ValueError(
                f"{where} reports lane {lane} done but its "
                f"info[{FINAL_OBS_KEY!r}] entry is None"
            )
        final_obs = check_obs(final, group, lanes=None,
                               where=f"{where} {FINAL_OBS_KEY}[{lane}]")
        # Under NEXT_STEP the final observation IS the one returned for that
        # lane. A mismatch means the returned observation was overwritten with
        # the new episode's first frame, which is SAME_STEP autoreset.
        returned = obs.select(lane) if obs.lanes is not None else obs
        if any(not np.array_equal(final_obs[leaf], returned[leaf])
               for leaf in group.names):
            raise ValueError(
                f"{where} lane {lane} ended, but info[{FINAL_OBS_KEY!r}][{lane}] "
                f"differs from the observation returned for that lane. The "
                f"contract is NEXT_STEP: the terminal step returns the final "
                f"observation itself and delivers the new episode's first "
                f"observation on the following step."
            )


def check_episode(info: dict[str, Any], *, where: str) -> None:
    """Assert the episode-statistics contract for a step that ended an episode."""
    episode = info.get(EPISODE_KEY) if isinstance(info, dict) else None
    if not isinstance(episode, dict) or not {"r", "l"} <= set(episode):
        raise ValueError(
            f"{where} ended an episode but info[{EPISODE_KEY!r}] is {episode!r}, "
            f"and evaluation reads the score from info[{EPISODE_KEY!r}]['r']. "
            f"ttrl.envs.SyncVectorEnv and ttrl.envs.AsyncVectorEnv accumulate "
            f"it for you; a VecEnv of your own must emit it per lane, and a "
            f"lane that transforms its own reward takes the return below that "
            f"transform with ttrl.envs.wrappers.RecordEpisodeStatistics."
        )
    for key in ("r", "l"):
        if not np.isfinite(episode[key]):
            raise ValueError(
                f"{where} reported info[{EPISODE_KEY!r}][{key!r}] "
                f"{episode[key]!r}. Accumulate the episode return and length "
                f"as finite plain numbers."
            )


def check_lane_episodes(info: dict[str, Any], done: np.ndarray, *, where: str) -> None:
    lanes = info.get(LANE_KEY) if isinstance(info, dict) else None
    if not isinstance(lanes, (list, tuple)) or len(lanes) != done.shape[0]:
        raise ValueError(
            f"{where} ended lanes {np.flatnonzero(done).tolist()} but "
            f"info[{LANE_KEY!r}] is not a list of {done.shape[0]} info dicts. "
            f"Pass each lane's own info through under {LANE_KEY!r}: that is "
            f"where an ended lane's episode statistics are read from."
        )
    for lane in np.flatnonzero(done).tolist():
        check_episode(lanes[lane], where=f"{where} lane {lane}")


def warn_if_swapped(ends: list[tuple[bool, int]], bound: int | None) -> None:
    """Warn when every episode ended as a truncation before the bound.

    THE TERMINATED/TRUNCATED SWAP, which is the one contract violation a
    driven rollout cannot refuse. An environment returning
    `(obs, r, truncated, terminated, info)` is a correct environment with two
    flags exchanged: every value has the right type, every episode ends, and
    the only damage is downstream, where a learner bootstraps the value of a
    state that has no successor instead of cutting it.

    A WARNING rather than a refusal, because truncating before the declared
    bound is legitimate: a `TimeLimit` counting inner steps under an action
    repeat truncates after fewer outer ones, and an author may end an episode
    early without calling it a termination. What is reported is the pattern
    the two cases share, so an author who meant it can read the sentence and
    move on while a swap is named.

    Episodes that end at the bound exactly are what a time limit does, so they
    raise nothing. An episode ending in `terminated` anywhere in the rollout
    proves the flag is reachable and clears the whole check.
    """
    if bound is None or not ends or any(terminated for terminated, _ in ends):
        return
    early = [length for _, length in ends if length < bound]
    if not early:
        return
    warnings.warn(
        f"the environment ended {len(ends)} episode(s) in this rollout, every "
        f"one of them with truncated=True and terminated=False, and "
        f"{len(early)} of them after {sorted(early)[:4]} step(s) while its "
        f"declared episode_bound is {bound}. The usual cause is `step` "
        f"returning its flags in the other order: the contract is "
        f"(obs, reward, terminated, truncated, info), where terminated means "
        f"the episode reached an end state of its own and truncated means it "
        f"was cut off with a future it still had. Swapping them trains a "
        f"learner that bootstraps the value of a terminal state, which no "
        f"later check can see. Return terminated=True for an episode that "
        f"ended on its own. If this environment really is only ever cut short, "
        f"which a TimeLimit counting inner steps under an action repeat is, "
        f"nothing here is wrong and this warning is the whole of it.",
        stacklevel=3,
    )
