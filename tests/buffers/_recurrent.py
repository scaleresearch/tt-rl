"""The pinned geometry every recurrent-rollout test uses, and one filled buffer."""

from __future__ import annotations

import numpy as np

from ttrl.buffers import RolloutBuffer
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

SEQ_LEN = 4
N_ENVS = 3
HIDDEN = 7
N_STEPS = 12
N_SEQ = 2
OBS_DIM = 5
N_ACTIONS = 6

#: (step, lane, terminated) for each episode end in the rollout.
#: t=5 lane 0 falls inside sequence 1 (steps 4..7); t=7 lane 1 is the last step
#: of sequence 1, so its reset lands on sequence 2's first step; t=11 lane 2 is
#: the last step of the rollout, so its reset lands on the next rollout's first
#: step; t=2 lane 0 is a TRUNCATION, which `dones` does not record.
BOUNDARIES = ((2, 0, False), (5, 0, True), (7, 1, True), (11, 2, True))

MEMORY = "actor"


def specs():
    return ObsGroup.of(ObsSpec.vector(OBS_DIM)), ActionSpec.discrete(N_ACTIONS)


def new_buffer(*, recurrent: bool = True, seed: int = 0) -> RolloutBuffer:
    obs_spec, action_spec = specs()
    return RolloutBuffer(
        obs_spec, action_spec, n_envs=N_ENVS, n_steps=N_STEPS, seed=seed,
        seq_len=SEQ_LEN if recurrent else None,
        hidden_sizes={MEMORY: HIDDEN} if recurrent else None,
    )


def boundary_flags(t: int, *, terminated_only: bool = False) -> np.ndarray:
    """The episode-end flags at step `t`, lane-major."""
    flags = np.zeros(N_ENVS, dtype=np.float32)
    for step, lane, terminated in BOUNDARIES:
        if step == t and (terminated or not terminated_only):
            flags[lane] = 1.0
    return flags


def fill(buffer: RolloutBuffer, *, boundary_source=boundary_flags,
         hidden_at=None) -> None:
    """One whole rollout of distinguishable values."""
    lanes = np.arange(N_ENVS, dtype=np.float32)
    for t in range(N_STEPS):
        tag = lanes + t / 100.0
        boundary = boundary_source(t)
        # The autoreset padding step: the step after a lane's boundary carries
        # a fabricated reward and an action the env ignored.
        valid = 1.0 - boundary_flags(t - 1)
        hidden = None
        if buffer.recurrent and t % SEQ_LEN == 0:
            chunk = t // SEQ_LEN
            rows = (hidden_at(chunk) if hidden_at is not None
                    else np.full((N_ENVS, HIDDEN), float(chunk))
                    + lanes.reshape(-1, 1) / 10.0)
            hidden = {MEMORY: rows}
        buffer.add(
            obs={"obs": np.tile(tag.reshape(-1, 1), (1, OBS_DIM))},
            actions=(np.arange(N_ENVS) + t) % N_ACTIONS,
            old_logp=-tag, values=tag * 2.0, rewards=tag,
            dones=boundary_flags(t, terminated_only=True),
            valid=valid,
            boundary=(boundary if buffer.recurrent else None),
            hidden=hidden,
        )
    buffer.compute_gae(0.99, 0.95, np.zeros(N_ENVS, dtype=np.float32))


def slot_tag(t: int, lane: int) -> float:
    """The value `fill` stored at `(t, lane)`, as float32 rounds it."""
    return np.float32(lane + t / 100.0)
