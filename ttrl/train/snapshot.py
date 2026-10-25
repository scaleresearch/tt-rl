"""What a run puts in a checkpoint, and where each piece comes from.

The write-side mirror of `ttrl.train.resume`. `ttrl.train.checkpoint.file` is
the serializer and knows nothing about replicas; `ttrl.train.trainer` owns the
run's resources and its loop. This is the one place that says which replica
each member is read off and which members a level carries, so the answer is
not spread across the loop.

EVERY MEMBER COMES OFF THE LEAD REPLICA. Each replica shares one estimator
across its own lanes but sees its own trajectories, so the replicas'
statistics differ by estimator noise; they estimate the same distribution, so
one replica's is a sound choice where a field-by-field mixture would not be.
`ttrl.train.resume` loads the lead's into every replica, as it does the
parameters.
"""

from __future__ import annotations

from ttrl.buffers.persist import owns_replay
from ttrl.train.checkpoint import L0, L1, save_checkpoint
from ttrl.train.checkpoint.tree import split
from ttrl.train.config import TrainConfig

__all__ = ["check_replay_support", "write_checkpoint"]


def check_replay_support(config: TrainConfig, agent) -> None:
    """Refuse `checkpoint_replay` for an algorithm that stores no experience.

    Asked once, when the run is built, rather than at the first checkpoint:
    that one lands `checkpoint_interval` frames in, and a setting that cannot
    be honoured must not be discovered half a million frames into a run.
    """
    if config.checkpoint_replay and not owns_replay(agent):
        raise ValueError(
            f"checkpoint_replay is on and algo {config.algo!r} stores no "
            f"replay buffer to checkpoint. It is an on-policy algorithm: its "
            f"rollout is consumed and discarded at every iteration, so a "
            f"resume has nothing to continue from and loses nothing by not "
            f"carrying it. Turn checkpoint_replay off, or set an off-policy "
            f"algo (one whose agent implements replay_state)."
        )


def write_checkpoint(replicas, config: TrainConfig, path: str, *, level: str,
                     frame: int, extra: dict | None = None) -> str:
    """Write one checkpoint of the lead replica and return its path.

    The level IS the question put to the agent: L0 is a snapshot to publish or
    score and asks for weights alone, L1 is resumable and asks for everything.
    Labelling everything L0 is what made the tier a lie.
    """
    agent = replicas.lead.agent
    save_checkpoint(
        path,
        agent_state=agent.state_dict(weights_only=level == L0),
        obs_spec=agent.obs_spec,
        action_spec=agent.action_spec,
        level=level,
        frame=frame,
        config=config,
        env_policy_state=replicas.lead.env.save_policy_state(),
        env_reward_state=(
            replicas.lead.env.save_run_state() if level == L1 else None),
        replay_state=_replay_member(config, agent, level),
        cadence_state=_cadence_member(replicas.lead, level),
        extra={"replicas": replicas.n, **(extra or {})},
    )
    return path


def _cadence_member(replica, level: str):
    """The replica's cadence counters, or None, which says there are none.

    None for the default run, where the cadence IS the agent and its update
    clock is already inside `agent_state`; storing it twice would give a
    resume two sources for one counter. A cadence injected separately
    (`ttrl.train.trainer.Trainer(cadence=...)`) holds those counters outside
    the agent, so a file that did not carry them would resume with the
    replay-ratio credit restarted at zero. One that declares no `state_dict`
    holds no counters and stores None, which is an answer; `ttrl.train.resume`
    is where the two answers are told apart.

    L1-only for the reason `env_reward_state` is: when a run learns is no part
    of what the weights compute.
    """
    cadence = replica.cadence
    if level != L1 or cadence is replica.agent:
        return None
    state_dict = getattr(cadence, "state_dict", None)
    if state_dict is None:
        return None
    state = state_dict()
    if not isinstance(state, dict):
        raise TypeError(
            f"{type(cadence).__name__}.state_dict() returned "
            f"{type(state).__name__}; a checkpoint stores it as a named tree "
            f"of numbers and arrays, so it must be a dict. Return the "
            f"cadence's counters by name, or drop state_dict entirely if the "
            f"cadence carries none."
        )
    # Checked HERE, before a byte is written, so the refusal names the
    # cadence whose credit would otherwise be lost rather than naming an npz
    # member a caller has never heard of.

    try:
        split(state, "cadence_state", {})
    except TypeError as exc:
        raise TypeError(
            f"{type(cadence).__name__}.state_dict() cannot be stored in a "
            f"checkpoint: {exc} This cadence holds the run's update credit "
            f"outside the agent, so a file written without it would resume "
            f"the run with that credit restarted at zero. Return the "
            f"counters as plain numbers and arrays."
        ) from exc
    return state


def _replay_member(config: TrainConfig, agent, level: str):
    """The agent's transitions, or None, which says the file carries none.

    OPT-IN and L1-only. Opt-in because a 1M-transition Atari buffer of
    stacked uint8 frames is 28 GB and a run checkpointing every 100k frames
    would fill a disk in an afternoon, so carrying it is a deliberate choice
    rather than a cost every run pays. L1-only for the reason
    `env_reward_state` is: experience is what a resume continues from and no
    part of what the weights compute, so nothing a deployment exports carries
    any of it.

    A file written without it does not pretend otherwise:
    `ttrl.train.resume` refuses a warm resume from one into an agent that
    owns a replay buffer, rather than continuing with an empty buffer behind
    an exploration schedule that has already annealed.
    """
    if level != L1 or not config.checkpoint_replay:
        return None
    return agent.replay_state()
