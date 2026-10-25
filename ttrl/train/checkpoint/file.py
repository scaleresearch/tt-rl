"""The npz on disk: one file, written atomically, readable with no hardware.

Two levels, and nothing below them needs a device, a replay buffer or an
environment:

  L0  the agent's own state, the specs it was built for, and the policy
      state its input transform is made of. A snapshot to publish or to
      score. Not resumable: no frame count, no run config.
  L1  L0 plus the frame count and the run's `TrainConfig`, so a resume needs
      the file and nothing else. The replay buffer is refilled by warm-up,
      which is a statistically indistinguishable continuation rather than the
      same trajectory.

The `ObsSpec` and `ActionSpec` are IN the file. They used to be recovered by
constructing a real environment just to read its action count, which made a
checkpoint unreadable without the environment suite installed and its assets
present. They are properties of the run, so the run records them.

`env_reward_state` is the third kind of state and an L1-ONLY member: the
reward normaliser's running return scale. A resume must continue it, because
a run whose loss changes scale on resume is not the run it reports itself as
continuing, and an L0 snapshot must not carry it, because the weights compute
the same function whatever the reward was divided by. The level system
already draws exactly that line, which is why this needed no new tier.

`cadence_state` is the fourth kind of state and L1-ONLY, because the cadence
is replaceable (`Trainer(cadence=...)`): an agent that is its own cadence
keeps its update clock in `agent_state` and this member stays None, while a
cadence injected separately holds the same counters outside the agent, and a
file without them resumes a run whose replay-ratio credit restarts at zero
while its frame count continues. `ttrl.train.resume` refuses that by name.

`replay_state` is the other L1-ONLY member, and the one the level comment
above used to be wrong about: a resume does NOT refill the buffer by warm-up
unless it is told to. The transitions are opt-in
(`TrainConfig.checkpoint_replay`) because an Atari buffer of uint8 frames is
tens of gigabytes, and a file without them refuses a warm resume rather than
continuing a run whose exploration schedule has annealed past the point where
a cold buffer can be refilled honestly. See `ttrl.buffers.persist`.

`env_policy_state` is a NAMED member rather than an entry in `extra`, at
both levels. It carries ONLY the state the weights are not interpretable
without: each transform-bearing wrapper's `POLICY_STATE` keys, such as
`NormalizeObservation`'s Welford mean, m2 and count. For a vector-observation
policy that transform IS part of the policy, and an artefact without it
receives inputs at the wrong scale (measured at 7.4x on one joint-angle
dimension of a Hopper policy). `extra` is the untyped bag a reader may
ignore; this decides what the weights compute, so it is declared.

`ttrl.envs.wrappers.core` owns the selection, via a `POLICY_STATE`
declaration per wrapper class, and `save_policy_state()` is what this format
is handed. Trajectory state is deliberately absent: an emulator core blob, a
frame stack, episode accumulators and MuJoCo qpos buy a bit-exact
mid-episode continuation, which this format explicitly does not promise. The
non-promise is not a licence to splice: a reader CLOSES the episode boundary
at the seam rather than reading across it, and `ttrl.train.resume` says what
that costs and where each piece of it happens. Keeping them out also keeps the
file free of opaque build-tied bytes, so every field in it has a name to put
in an error message. Anyone wanting bit-exact resume should write a separate
opt-in artefact beside the checkpoint rather than change what this one means.

`TrainConfig.device` is NOT in the file. Which chip a run happened to sit on,
and how large its L1 scratch was, say nothing about the weights, so a reader
of a checkpoint has no business inheriting them: see
`TrainConfig.UNSAVED_FIELDS`. `ttrl.train.checkpoint.tree` carries both that
rule and the field-name contract it is part of.

Format: a single `.npz` holding every array plus one JSON `__meta__` member.
Arrays live exactly once, as plain npz arrays; meta carries only counters,
hyperparameters and structure. There is no `pickle` anywhere, so loading a
checkpoint cannot execute arbitrary code.

Atomicity: `ttrl.train.checkpoint.atomic`, so a crash never leaves a
half-written file where a reader looks.

THE FORMAT IS PRE-RELEASE AND HAS NO MIGRATIONS. `FORMAT` is written into
every file and compared for equality; a file carrying anything else is refused
by name. The marker is an opaque token rather than a number so that no reader
can grow a range over it, and the only reason to keep it at all is that it
turns an unintelligible decode error into a sentence.
"""

from __future__ import annotations

import dataclasses
import io
import json
from typing import Any

import numpy as np

from ttrl.spaces import ActionSpec, ObsGroup
from ttrl.train.checkpoint.atomic import atomic_write_bytes
from ttrl.train.checkpoint.specs import json_default
from ttrl.train.checkpoint.tree import split

__all__ = ["FORMAT", "L0", "L1", "LEVELS", "save_checkpoint"]

#: The on-disk format marker. Compared for EQUALITY and nothing else. Replace
#: it with a fresh token whenever the layout changes; every file written before
#: that is then refused, which is what a pre-release format with no migrations
#: means.
FORMAT = "ttrl-checkpoint-9c42ad"

L0 = "L0"
L1 = "L1"
LEVELS = (L0, L1)

_ARRAY_PREFIX = "state"


def save_checkpoint(
    path: str,
    *,
    agent_state: dict,
    obs_spec: ObsGroup,
    action_spec: ActionSpec,
    level: str = L1,
    frame: int = 0,
    config=None,
    env_policy_state: list | None = None,
    env_reward_state: list | None = None,
    replay_state: dict | None = None,
    cadence_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint atomically at the requested level.

    `env_policy_state` is one `Env.save_policy_state()`: the transform state
    the weights are not interpretable without. A caller that has an env
    passes it, and `Trainer` always does. It is NOT required here: this
    function is the serializer, and making the keyword mandatory put a
    guarantee about one caller's completeness into the one place that cannot
    tell a caller who forgot from an env that genuinely has none. The
    guarantee lives where the harm is instead, in
    `ttrl.train.resume.resume_replicas`, which refuses a file that carries
    none. An env whose stack holds no transform state snapshots as `[]`,
    which is an answer and not an absence.
    """
    if level not in LEVELS:
        raise ValueError(
            f"level must be one of {LEVELS}, got {level!r}. The trajectory-exact "
            "tier was removed: a resume refills the buffer by warm-up."
        )
    if level == L0 and replay_state:
        raise ValueError(
            "an L0 checkpoint is a weights snapshot and carries no run state, "
            "but replay_state holds a buffer of transitions. Experience is "
            "what a resume continues from and is no part of what the weights "
            "compute, so it belongs to an L1 file and to nothing a deployment "
            "exports."
        )
    if level == L0 and env_reward_state:
        raise ValueError(
            f"an L0 checkpoint is a weights snapshot and carries no run state, "
            f"but env_reward_state names {[n for n, _ in env_reward_state]}. A "
            f"reward normaliser's running scale is not part of what the "
            f"weights compute, so it belongs to an L1 file and to nothing a "
            f"deployment exports."
        )
    if level == L0 and cadence_state:
        raise ValueError(
            "an L0 checkpoint is a weights snapshot and carries no run state, "
            "but cadence_state holds an update cadence's counters. When a run "
            "learns is no part of what the weights compute, so it belongs to "
            "an L1 file alone."
        )
    if level == L1 and config is None:
        raise ValueError(
            "an L1 checkpoint carries the run config so a resume needs nothing "
            "else; pass config=, or write level='L0' for a weights snapshot."
        )

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "format": FORMAT,
        "level": level,
        "obs_spec": dataclasses.asdict(ObsGroup.of(obs_spec)),
        "action_spec": dataclasses.asdict(action_spec),
        "agent_state": split(agent_state, _ARRAY_PREFIX, arrays),
        "env_policy_state": (
            None if env_policy_state is None
            else split(env_policy_state, "env_policy_state", arrays)
        ),
        "extra": extra or {},
    }
    if level == L1:
        # Through `split`, like the agent's state, so the config comes back as
        # itself: its `algo_cfg` is the algorithm's own frozen dataclass and an
        # agent rebuilt from a dict-shaped stand-in would not accept it.
        meta["frame"] = int(frame)
        meta["config"] = split(config, "config", arrays)
        # L1 only, and None rather than [] when the caller has nothing to
        # say, for the reason `env_policy_state` is: a reader must be able to
        # tell a file that carries no run state from a run that had none.
        meta["env_reward_state"] = (
            None if env_reward_state is None
            else split(env_reward_state, "env_reward_state", arrays)
        )
        # None rather than {} for the same reason: a reader must be able to
        # tell a file that carries no transitions from a run that had none.
        meta["replay_state"] = (
            None if replay_state is None
            else split(replay_state, "replay_state", arrays)
        )
        # None for a run whose agent is its own cadence: those counters are
        # in `agent_state` already.
        meta["cadence_state"] = (
            None if cadence_state is None
            else split(cadence_state, "cadence_state", arrays)
        )

    buf = io.BytesIO()
    payload = dict(arrays)
    blob = json.dumps(meta, default=json_default).encode("utf-8")
    payload["__meta__"] = np.frombuffer(blob, dtype=np.uint8)
    np.savez(buf, **payload)
    atomic_write_bytes(path, buf.getvalue())
