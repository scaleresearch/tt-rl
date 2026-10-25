"""The frozen observation transform an artifact carries, from a checkpoint.

Split from `ttrl.deploy.export` because it is the one part of an artifact that
is not about the policy at all: it is the environment stack's, and it is the
part whose absence is invisible. A running normaliser's mean and variance are
applied to every observation the network ever saw, so weights exported without
them command the wrong action from a correct-looking file.

A REWARD normaliser is deliberately not here. Its running scale is `RUN_STATE`
rather than `POLICY_STATE` (`ttrl.envs.wrappers.core`), so it never reaches an
L0 snapshot and never reaches this function: a deployed policy consumes no
reward, and the weights compute the same function whatever the training reward
was divided by.
"""

from __future__ import annotations

import numpy as np

from ttrl.envs.vec_normalize import FrozenNormalization

__all__ = ["NORMALIZER", "observation_transform"]

#: The one wrapper whose policy state this format carries. A stack holding
#: any other transform-bearing layer is refused by name: a transform nobody
#: reimplemented here is a transform the artifact would silently omit.
#:
#: It is `ttrl.envs.vec_normalize.NormalizeObservation`, which sits above the
#: vector environment and holds ONE estimator over every lane, so the
#: statistics written here are the ones the whole run was fitted on. They used
#: to come from a per-lane wrapper, which made the exported transform whichever
#: lane the lead replica happened to hold. The name and the key set are
#: unchanged by that move, which is why a file written before it still
#: exports: its statistics really were the transform those weights were
#: trained under, lane-local as they were.
NORMALIZER = "NormalizeObservation"


def actor_columns(stats: np.ndarray, obs_spec) -> np.ndarray:
    """`stats`, one entry per observation column, cut to the actor's columns."""
    if not obs_spec.privileged:
        return stats
    return np.concatenate([stats[a:b] for a, b in obs_spec.actor_spans])


def observation_transform(policy_state, path: str, arrays: dict,
                          obs_spec) -> dict | None:
    """The transform's description, with its statistics written into `arrays`.

    The transform is REQUIRED to be present as an answer. A checkpoint that
    does not say what transform was applied is refused; an empty policy state
    is an answer ("this stack applied none") and returns None.
    """
    if policy_state is None:
        raise ValueError(
            f"cannot export a deployable policy: {path!r} carries no "
            f"'env_policy_state', so it does not say what transform its "
            f"observations went through. Every run under this version of ttrl "
            f"stores that member, and the format is pre-release and carries "
            f"no migrations, so nothing can add it to this file. A running "
            f"normaliser's mean and variance are applied to every observation "
            f"the network ever saw, so weights exported without them command "
            f"the wrong action from a correct-looking artifact. Re-save the "
            f"checkpoint from a run under this version of ttrl."
        )
    names = [name for name, _ in policy_state]
    unknown = [n for n in names if n != NORMALIZER]
    if unknown:
        raise ValueError(
            f"cannot export a deployable policy: {path!r} was trained under the "
            f"transform-bearing wrapper(s) {unknown}, which this artifact format "
            f"does not reimplement in numpy. Dropping them would export weights "
            f"that receive inputs they were never fitted on. Add them to "
            f"ttrl.deploy, or train under a stack this exporter covers."
        )
    if not names:
        return None
    if len(names) != 1:
        raise ValueError(
            f"cannot export a deployable policy: {path!r} applies {len(names)} "
            f"observation normalisers in one stack, and an artifact carries one "
            f"mean and one deviation. Whitening a whitened observation leaves "
            f"statistics neither layer fitted, so the two cannot be composed "
            f"here. Train under a stack with a single NormalizeObservation: "
            f"env.normalize_obs applies that one above the vector environment, "
            f"so drop any copy added by hand."
        )
    # The transform itself, with the `count <= 1` variance placeholder and the
    # epsilon under the root taken from the wrapper that fitted it rather than
    # recomputed here. The artifact stores the resulting standard deviation so
    # a deployed step does no statistics.
    items = dict(policy_state[0][1])
    frozen = FrozenNormalization(items)
    mean, std, count = frozen.mean, frozen.std, frozen.count
    if mean.shape != obs_spec.shape:
        raise ValueError(
            f"cannot export a deployable policy: {path!r} carries normaliser "
            f"statistics of shape {mean.shape} for an observation of shape "
            f"{obs_spec.shape}."
        )
    arrays["obs_mean"] = actor_columns(mean, obs_spec)
    arrays["obs_std"] = actor_columns(std, obs_spec)
    return {
        "wrapper": NORMALIZER,
        "clip": frozen.clip,
        "epsilon": float(items["epsilon"]),
        "count": count,
    }
