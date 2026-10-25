"""The env-shaped half of what a warm resume demands of a checkpoint.

Split from `ttrl.train.resume`, which orchestrates a resume and owns the
refusals about the agent's own carried state (its replay buffer, its update
cadence) and about config fields the file never recorded. Everything here is
about the ENVIRONMENT: the transform state a policy is not interpretable
without, the running statistics a run accumulated, and the stored env config
that is the only other evidence of which wrappers the training stack had.
`ttrl.train.resume` is where the argument for refusing rather than defaulting
any of it is written down.
"""

from __future__ import annotations

from ttrl.train.config import TrainConfig

__all__ = ["refuse_an_unrecorded_transform", "restore_env", "restore_run_state"]


def restore_run_state(replica, loaded: dict, path: str) -> None:
    """Continue the run statistics that are not part of what the weights do.

    A reward normaliser's running return scale. The LIVE stack is what says
    whether there is any to restore, exactly as `refuse_an_unrecorded_transform`
    asks the live stack rather than the file: a file carries the member as None
    when its run normalised no reward, so a stack that has one is told that the
    file cannot supply it rather than resumed with the scale restarted at zero.
    """
    stored = loaded.get("env_reward_state")
    live = [name for name, _ in replica.env.save_run_state()]
    if stored is None:
        if not live:
            return
        raise ValueError(
            f"{path!r} carries no 'env_reward_state' and this run's "
            f"environment carries run statistics for {live}, so the file was "
            f"written by a run that normalised no reward. Resuming "
            f"anyway would restart the reward scale at zero, which changes "
            f"the size of every gradient the resumed run takes while "
            f"reporting itself as a continuation. Start the run afresh, or "
            f"resume with env.normalize_reward off."
        )
    replica.env.load_run_state(stored)


def _wrapper_identity(config: TrainConfig) -> list[tuple[str, tuple]]:
    """What decides which wrappers a config's env stacks are built from.

    The builder's registered name and its keyword arguments, plus the
    `normalize_*` fields, for the training env and the eval env. The
    normalisers are stated on the `EnvConfig` rather than passed to the
    builder, because each is one estimator above the whole vector env, so
    they are part of what the stack IS and not of what it is run with.
    Everything else in an `EnvConfig` (the seed, the lane count, the
    sync/async choice) is the latter and cannot add or remove a layer.
    """
    return [
        (env.name, tuple(sorted(env.kwargs.items())),
         env.normalize_obs, env.normalize_obs_clip,
         env.normalize_reward, env.normalize_reward_clip,
         env.normalize_reward_gamma)
        for env in (config.env, config.eval_env)
        if env is not None
    ]


def refuse_an_unrecorded_transform(replicas, path: str, stored: TrainConfig,
                                    live: TrainConfig) -> None:
    """A file carrying no `env_policy_state`, judged by what this env carries.

    `env_policy_state is None` says only that the writer passed no snapshot.
    It says nothing about whether there is any state to lose, and for most
    runs there is not. An Atari stack
    declares `POLICY_STATE = ()` on every layer, so it carries no cross-episode
    statistics at all; a checkpoint written by that stack stores an EMPTY
    snapshot and resumes, and refusing the same env's snapshot-free file on
    that alone refused a resume that had provably nothing to restore. That
    cost a real 250,000 frames.

    So the question asked here is the one the refusal is actually about: does
    THIS run's env carry transform state? Every replica is asked, because each
    builds its own env and a refusal must name the stack it is about. A stack
    that carries state still refuses, and must: resuming a normalised policy
    against statistics restarted at zero feeds the network observations on a
    scale it never saw, measured at 4.2% of the score over 250-step Hopper
    episodes.

    The second refusal below covers what a bearer-free stack still cannot
    know, which is what the TRAINING stack carried. A file with a snapshot says
    so and is checked by `restore_policy_state`; one without does not, and the
    two env configs are then the only evidence left.
    """
    bearers = {
        name
        for replica in replicas.replicas
        for name, _items in replica.env.save_policy_state()
    }
    if not bearers:
        # This stack carries nothing, but the file does not say what the
        # TRAINING stack carried, and a modern file would: a non-empty
        # snapshot loaded into a bearer-free stack is refused by
        # `restore_policy_state`. The stored env config is the only other
        # evidence, so a stack built the same way was this stack and the
        # resume is provably safe, while one built differently may apply a
        # different transform with nothing able to tell.
        #
        # Compared on what SELECTS the wrappers and nothing else. A seed, a
        # lane count and the sync/async choice cannot add or remove a
        # transform-bearing layer, and refusing a resume that changed one of
        # them would be the same false refusal this replaced.
        if _wrapper_identity(stored) == _wrapper_identity(live):
            return
        raise ValueError(
            f"{path!r} carries no 'env_policy_state' and was written under a "
            f"different environment config than this run uses "
            f"({stored.env} then, {live.env} now). This run's environment "
            f"carries no transform state, but the file cannot say whether the "
            f"training environment did, so nothing here can tell whether the "
            f"weights were trained on a scale this stack does not apply. "
            f"The checkpoint format is pre-release and carries no "
            f"migrations, so a file that does not store that member cannot "
            f"be made to. "
            f"Resume under the config the checkpoint stores "
            f"(ttrl.config_from_checkpoint prints it), score the weights with "
            f"ttrl.evaluate_checkpoint, or start the run afresh."
        )
    raise ValueError(
        f"{path!r} carries no 'env_policy_state', and this run's environment "
        f"carries policy state for {sorted(bearers)}, so there is state the "
        f"file cannot supply. Every run under this version of ttrl stores "
        f"that member, and the format is pre-release and carries no "
        f"migrations, so nothing can add it to this file after the fact; "
        f"resuming anyway would restart those statistics at zero and so "
        f"change the function the loaded weights compute. Score its weights "
        f"with ttrl.evaluate_checkpoint, which needs no transform state, or "
        f"start the run afresh. An environment whose layers all declare "
        f"POLICY_STATE = () resumes from a file without that member without "
        f"any of this, because it has nothing to restore."
    )


def restore_env(replica, policy_state: list, path: str) -> None:
    """Apply a checkpoint's policy state, or say why this env cannot take it.

    `restore_policy_state` already refuses a stack whose transform-bearing
    layers are not the checkpoint's. What it cannot say is which file the
    state came from or what to do. `ValueError` alone is caught: this member
    holds named floats and small arrays, with no opaque blob left for an
    environment suite to fail to parse.
    """
    try:
        replica.env.load_policy_state(policy_state)
    except ValueError as exc:
        raise ValueError(
            f"{path!r} holds policy state this run's env cannot take: {exc} "
            "The checkpoint's env config and this run's differ in a wrapper "
            "that transforms observations, so the stored statistics describe a "
            "transform this stack does not apply. Resume with the env config "
            "the checkpoint stores (ttrl.config_from_checkpoint prints it), or "
            "start a new run under the config you want."
        ) from exc
