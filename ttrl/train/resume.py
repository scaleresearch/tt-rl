"""What a warm resume demands of a checkpoint, and what it refuses.

A resume promises to continue the checkpointed run rather than a similar one.
Everything here is one application of that: a field the reader had to default,
an environment transform the file cannot supply, a run statistic it does not
carry, a replay buffer it never stored. Each is refused BY NAME with the
replacements named, because every one of them otherwise produces a run that
trains happily and reports itself as a continuation of something it is not.

WHAT A RESUME DOES NOT PROMISE, and what it does about it. No part of a
Gymnasium simulator's state is in a checkpoint and nothing can put it there
from outside the env, so `ttrl.train.loop.run_loop` resets every replica and
the episode a lane was halfway through does not continue. That is a
documented non-promise (`ttrl.train.checkpoint.file`), and on its own it was
still a defect: the restored replay, rollout, advantage chain and recurrent
carry all read ACROSS the reset, so every mid-episode resume spliced the two
episodes together and trained on the join. So each piece of carried state
closes its own seam where it is restored, and a boundary is the one thing a
resume does add to what the file says:
`ttrl.buffers.persist` ends each lane's unfinished episode against the
successor the writer recorded, `ttrl.algos.ppo.partial` turns the last stored
rollout step into a padding slot and marks the next step an episode start,
and `ttrl.algos.base.AutoresetTracker` starts its phase fresh. Nothing
stored, trained on or carried crosses the join afterwards.

`ttrl.train.resume_env` holds the env-shaped refusals this orchestrates.
`ttrl.train.restore` is the other half, rebuilding an agent from a file, and
is what a score and a deployment export ask. `ttrl.train.snapshot` is the
write-side mirror of this module.
"""

from __future__ import annotations

from ttrl.buffers.persist import owns_replay
from ttrl.train.checkpoint import L1, load_checkpoint
from ttrl.train.config import TrainConfig
from ttrl.train.resume_env import (
    refuse_an_unrecorded_transform, restore_env, restore_run_state,
)

__all__ = ["resume_replicas"]


def resume_replicas(replicas, path: str, config: TrainConfig) -> int:
    """Load an L1 checkpoint into every replica and return its frame count.

    Every replica takes the IDENTICAL state. Parameters are shared across
    replicas by construction of the averaging, so one checkpointed set is the
    only well-defined thing to resume into, and the environment state goes
    the same way rather than being left to restart per replica.

    `config` is the config THIS run was built from, which the checks below
    need: a file that carries no policy-state snapshot gives nothing to
    compare a stack against, and the two configs are then the only evidence
    of whether the transform changed.
    """
    loaded = load_checkpoint(path)
    if loaded["level"] != L1:
        raise ValueError(
            f"{path!r} is a {loaded['level']} checkpoint and carries no frame "
            "count, so a run cannot continue from it. Score it with "
            "ttrl.evaluate_checkpoint, or resume from an L1 file."
        )
    policy_state = loaded["env_policy_state"]
    if policy_state is None:
        refuse_an_unrecorded_transform(
            replicas, path, loaded["config"], config)
        # Proven above to be a stack with no transform-bearing layer under the
        # env config that wrote the file, and the empty snapshot is exactly
        # what such a stack stores today. Passing it rather than branching at
        # the call site keeps `restore_policy_state` the one path.
        policy_state = []
    _refuse_defaulted_config(loaded, path)
    for replica in replicas.replicas:
        replica.agent.load_state_dict(loaded["agent_state"])
        restore_env(replica, policy_state, path)
        restore_run_state(replica, loaded, path)
        _restore_replay(replica, loaded, path)
        _restore_cadence(replica, loaded, path)
    return int(loaded["frame"])


def _restore_replay(replica, loaded: dict, path: str) -> None:
    """Continue the experience the run had collected, or refuse by name.

    THE DEFAULT PATH MUST NOT LIE. A replay buffer is opt-in
    (`TrainConfig.checkpoint_replay`, see `ttrl.train.snapshot`) because an
    Atari buffer is tens of gigabytes. What is not optional is saying so: a
    file carrying no transitions resumed into an off-policy agent gives it an
    EMPTY buffer while `frame_count`, the epsilon schedule and the update
    clock all continue from where they stopped, so the first thousands of
    gradient steps after the resume train on a nearly empty buffer at a
    fully annealed exploration rate. That ruins the run and raises nothing,
    which is worse than failing.

    Asked of the LIVE agent, exactly as `restore_run_state` asks the live
    env: an on-policy agent owns no replay buffer, loses nothing, and resumes
    from any file without any of this.

    There is deliberately NO flag that resumes an off-policy run cold. Such a
    flag would have exactly one effect, the silently worse policy above, and
    both honest answers to wanting it are already here: score the weights, or
    start afresh.
    """
    if not owns_replay(replica.agent):
        return
    stored = loaded.get("replay_state")
    if stored is None:
        raise ValueError(
            f"{path!r} carries no 'replay_state' and this run's agent owns a "
            f"replay buffer. Resuming would hand it an EMPTY buffer while the "
            f"frame count, the exploration schedule and the update clock "
            f"continue from frame {loaded['frame']}, so the resumed run would "
            f"take its next thousands of gradient steps on almost no data at "
            f"an exploration rate that has already annealed, and would report "
            f"itself as a continuation while converging somewhere worse. "
            f"A checkpoint stores transitions only when the run that wrote it "
            f"set checkpoint_replay. Resume from a file written with "
            f"checkpoint_replay on, score these weights with "
            f"ttrl.evaluate_checkpoint, which needs no experience, or start "
            f"the run afresh."
        )
    replica.agent.load_replay_state(stored)


def _restore_cadence(replica, loaded: dict, path: str) -> None:
    """Continue the replay-ratio credit of a cadence that lives outside the agent.

    An agent that is its own cadence keeps its update clock inside its own
    state and is already restored above; there is nothing here for the
    default run. A cadence injected separately
    (`ttrl.train.trainer.Trainer(cadence=...)`) holds the same counters
    outside the agent, and they are run state: a resume that dropped them
    would restart the credit at zero while the frame count, the exploration
    schedule and every other clock continue, so the resumed run takes fewer
    gradient steps per frame than it is configured for and reports itself as
    a continuation. The LIVE cadence is what says whether there is any to
    restore, exactly as `restore_run_state` asks the live env.

    Both mismatches are refused BY NAME, because each is credit silently
    changed: a stateful cadence against a file that carries none, and a file
    that carries some against a cadence that cannot take it.
    """
    cadence = replica.cadence
    stored = loaded.get("cadence_state")
    if cadence is replica.agent:
        if stored is None:
            return
        raise ValueError(
            f"{path!r} carries 'cadence_state' and this run passed no "
            f"cadence=, so its agent is its own cadence and the stored "
            f"counters have nowhere to go. The file was written by a run "
            f"whose update cadence lived outside the agent, and resuming "
            f"without it would continue the frame count while restarting "
            f"that cadence's credit at zero. Resume with the same cadence= "
            f"factory the run was trained with, score these weights with "
            f"ttrl.evaluate_checkpoint, which needs no credit, or start the "
            f"run afresh."
        )
    load = getattr(cadence, "load_state_dict", None)
    who = type(cadence).__name__
    if stored is None:
        if load is None or getattr(cadence, "state_dict", None) is None:
            return
        raise ValueError(
            f"{path!r} carries no 'cadence_state' and this run's cadence "
            f"{who} declares state_dict/load_state_dict, so it holds counters "
            f"the file cannot supply. A checkpoint stores them for every run "
            f"whose cadence declares both, and the format is pre-release and "
            f"carries no migrations, so nothing can add the member to this "
            f"file after the fact; resuming anyway would restart this "
            f"cadence's replay-ratio credit at zero while the frame count "
            f"continues from {loaded['frame']}. Resume from a file written "
            f"with this cadence, score these weights with "
            f"ttrl.evaluate_checkpoint, or start the run afresh. A cadence "
            f"that declares neither method holds no credit and resumes from "
            f"any file without any of this."
        )
    if load is None:
        raise ValueError(
            f"{path!r} carries 'cadence_state' and this run's cadence {who} "
            f"has no load_state_dict, so the stored counters cannot be put "
            f"back and the resumed run would restart its update credit at "
            f"zero while continuing from frame {loaded['frame']}. Give {who} "
            f"a load_state_dict taking what its state_dict returns, or "
            f"resume with the cadence the file was written with."
        )
    load(stored)


def _refuse_defaulted_config(loaded: dict, path: str) -> None:
    """A warm resume must continue the checkpointed run, not a similar one.

    A config field added since the file was written reads back as today's
    default, which is right for a score and wrong for a resume: the run would
    carry a setting nothing recorded and nothing printed. The same response as
    a missing `env_policy_state`, for the same reason.
    """
    absent = loaded.get("defaulted_fields") or []
    if absent:
        raise ValueError(
            f"{path!r} carries no value for {absent}, so this reader supplied "
            f"them from today's defaults. A resume continues the checkpointed "
            f"run, and continuing it under settings the file never recorded "
            f"would change what the run is without saying so. A score is "
            f"refused for the same reason (`ttrl.train.restore` says why a "
            f"weights-only read is not config-free), so pass the config the "
            f"run was trained from, which replaces the stored one entirely, "
            f"or start the run afresh."
        )
