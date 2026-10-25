"""Checkpoint in, deployable policy artifact out.

What a roboticist needs off a training box is the actor alone: the trunk, the
head, the action map and the observation transform the weights were fitted
under. A checkpoint carries all of that plus the critics, the Polyak targets,
three optimizers' moments and the replay clock, and reading it needs
`ttrl.spaces` at least. This writes the actor out as a file
`ttrl.deploy.read` and `ttrl.deploy.artifact` read with numpy alone.

EVERY ALGORITHM IN THE REGISTRY EXPORTS. Which member of the agent state holds
the actor, and which map turns its head's row into an action, is one table,
`ttrl.deploy.policies.SOURCES`, with an entry per algorithm in
`ttrl.algos.registry.ALGOS`. The discrete and continuous action kinds are
covered: a tanh-squashed
Gaussian (SAC), an unsquashed Gaussian clipped to the bounds (PPO continuous)
and a greedy argmax over logits or Q values (PPO discrete, DQN).

Scope, and why each limit is a refusal rather than a branch:

  One observation leaf. A vector leaf feeds the trunk directly. An image
  leaf goes through its conv tower first, whose strides are read off the
  encoder the run named (`ttrl.deploy.towers`), so a custom image encoder is
  refused by name. An image policy carries no stored normaliser and no mask.

  No composite action. A multi_discrete policy's greedy action is a
  per-factor argmax over segments of one head, and every map here reads the
  head as one distribution. `SOURCES` has no entry for the kind, so the
  combination is refused by name rather than argmaxed over the concatenation,
  which would return a single index into logits that are not one
  distribution.

  No recurrent policy. The artifact is one feedforward map with no carried
  state, so a checkpoint holding a memory (PPO with `recurrent` set) is
  refused rather than exported without its GRU.

  No masked policy over a continuous action. A legal-action mask names
  members of a finite action set, so the mask is carried for the `argmax` map
  and refused for the others. A mask itself is NOT a limit: it is the final
  `ObsSpec.action_mask` entries of the observation the actor already reads, so
  the artifact stores its width, slices it off its one input and drops the
  illegal columns before the argmax. No second input, and no defaulting to
  all-legal, which is the fallback that silently commands an illegal action.

  No whitened mask. A stored observation normaliser would rewrite the mask's
  indicators into whatever its statistics say, so the combination is refused.
  `NormalizeObservation` refuses a masked observation for the same reason, and
  a checkpoint carrying both was not written by this library's stack.

The artifact's input is the ACTOR's observation, `ObsSpec.actor`: a
critic-only field is privileged simulator state the real robot cannot sense,
so an artifact that asked for it would be undeployable by construction. The
normaliser whitens each column on its own, so cutting its statistics down to
the actor's spans gives the same numbers as whitening the whole observation
and cutting afterwards. Which width the actor really reads is taken from the
first trunk weight and checked against the spec, so a spec that says one
thing and weights that say another is named rather than exported.
"""

from __future__ import annotations

import io
import json

import numpy as np

from ttrl.deploy.artifact import ACTION_MAPS
from ttrl.deploy.read import ARTIFACT_FORMAT, META_KEY
from ttrl.deploy.policies import source_for
from ttrl.deploy.towers import split_tower
from ttrl.deploy.transform import observation_transform
from ttrl.train.checkpoint.atomic import atomic_write_bytes
from ttrl.train.checkpoint.read import load_checkpoint

__all__ = ["export_policy"]


def export_policy(checkpoint_path: str, out_path: str) -> str:
    """Write the actor of `checkpoint_path` to `out_path`, and return it.

    The result is read by `ttrl.load_policy` on a host with numpy and nothing
    else, and maps a raw environment observation to the deterministic action:
    a continuous one in the environment's own units, a discrete one as the
    chosen action's index. Works on an L0 or an L1 checkpoint: every number
    the artifact needs describes the weights, so none of it comes from the run
    config. `ttrl.export_onnx` turns the result into an ONNX graph.
    """
    loaded = load_checkpoint(checkpoint_path)
    obs_group, action_spec = loaded["obs_spec"], loaded["action_spec"]
    leaf, obs_spec = _one_leaf(obs_group, checkpoint_path)
    state = loaded["agent_state"]
    source = source_for(state.get("config"), checkpoint_path)
    kind = action_spec.kind
    _require(kind in source.maps, checkpoint_path,
             f"was trained with {source.algo} on a {kind} action spec, and this "
             f"exporter writes {source.algo}'s {sorted(source.maps)} policy. "
             f"That combination has no action map, so its head's output is not "
             f"an action this format can compute")
    action_map = source.maps[kind]

    trunk, head = _actor(state, source, kind, checkpoint_path)
    stages, geometry, trunk = split_tower(trunk, obs_spec, leaf, state["config"],
                                          checkpoint_path, _require)
    actor_spec = obs_spec.actor
    _require(bool(stages) or trunk[0][0].shape[0] == actor_spec.size,
             checkpoint_path,
             f"has an actor trunk reading {trunk[0][0].shape[0]} inputs while its "
             f"ObsSpec gives the actor {actor_spec.size} of {obs_spec.size} "
             f"entries (critic-only field(s) {list(obs_spec.privileged)}). The "
             f"file does not say which columns the weights were fitted on")
    mask = int(actor_spec.action_mask)
    if mask:
        _require(action_map == "argmax", checkpoint_path,
                 f"carries a {mask}-wide legal-action mask in its observation "
                 f"while its action map is {action_map!r}, which chooses no "
                 f"column for the mask to name. A mask names members of a "
                 f"finite action set, so train this environment with a discrete "
                 f"ActionSpec, or drop action_mask from the ObsSpec")
        _require(mask == action_spec.n, checkpoint_path,
                 f"carries a {mask}-wide legal-action mask for an action spec "
                 f"of n={action_spec.n}. The mask holds one entry per action, so "
                 f"build the ObsSpec with action_mask={action_spec.n}")
    width = head[0].shape[1]
    _require(width == action_spec.width, checkpoint_path,
             f"has a policy head emitting {width} numbers for an action spec of "
             f"width {action_spec.width}")
    # `ScaledLinear` folds `ObsSpec.scale` into the first weight rather than
    # multiplying every observation, so the artifact folds it the same way and
    # the deployed forward is the device forward op for op.
    # An image tower folded it into its first conv already.
    if not stages:
        first_weight, first_bias = trunk[0]
        trunk[0] = (first_weight * np.float32(obs_spec.scale), first_bias)

    arrays: dict[str, np.ndarray] = {}
    for index, (weight, bias) in enumerate(stages):
        arrays[f"conv.{index}.weight"] = weight
        arrays[f"conv.{index}.bias"] = bias
    for index, (weight, bias) in enumerate(trunk):
        arrays[f"trunk.{index}.weight"] = weight
        arrays[f"trunk.{index}.bias"] = bias
    arrays["head.weight"], arrays["head.bias"] = head
    if ACTION_MAPS[action_map]:
        arrays["action_low"] = action_spec.low_array.astype(np.float32)
        arrays["action_high"] = action_spec.high_array.astype(np.float32)

    normalizer = observation_transform(
        loaded["env_policy_state"], checkpoint_path, arrays, obs_spec)
    _require(not (mask and normalizer), checkpoint_path,
             f"carries a {mask}-wide legal-action mask in its observation and a "
             f"stored observation normaliser, which would whiten the mask's own "
             f"indicators out of {{0, 1}}. Normalise the features inside the "
             f"environment and append the mask afterwards, which is what "
             f"NormalizeObservation itself asks for, and train again")
    _require(not (stages and (mask or normalizer)), checkpoint_path,
             "is an image policy with a stored observation normaliser or a "
             "legal-action mask, and this format applies both to vector "
             "observations only")
    meta = {
        "artifact_format": ARTIFACT_FORMAT,
        "algo": source.algo,
        "action_kind": kind,
        "action_map": action_map,
        "obs_dim": None if stages else int(actor_spec.size),
        "obs_shape": [int(d) for d in actor_spec.shape],
        "conv": geometry,
        "privileged_fields": list(obs_spec.privileged),
        "action_dim": None if action_spec.dim is None else int(action_spec.dim),
        "n_actions": None if action_spec.n is None else int(action_spec.n),
        "action_mask": mask,
        "trunk_layers": len(trunk),
        "normalizer": normalizer,
        "source_checkpoint": checkpoint_path,
    }
    buffer = io.BytesIO()
    payload = dict(arrays)
    payload[META_KEY] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(buffer, **payload)
    atomic_write_bytes(out_path, buffer.getvalue())
    return out_path


def _require(condition: bool, path: str, complaint: str) -> None:
    if not condition:
        raise ValueError(f"cannot export a deployable policy: {path!r} {complaint}.")


def _one_leaf(obs_group, path: str):
    """`(name, spec)` of the single leaf this artifact takes as its input."""
    _require(len(obs_group) == 1, path,
             f"was trained on an observation of {len(obs_group)} leaves "
             f"{list(obs_group.names)}, and this artifact's input is one "
             f"observation vector. A multi-modal policy needs an artifact "
             f"format that carries one input per leaf")
    name = obs_group.names[0]
    return name, obs_group[name]


def _actor(state: dict, source, kind: str, path: str) -> tuple[list, tuple]:
    """The actor's `Linear` weights: its trunk in order, and its head.

    Found by name rather than by rebuilding the networks, which would need a
    device. The pattern is anchored at the actor's own member, so the critics,
    their targets and DQN's lagged copy cannot match it.
    """
    nets = state.get(source.state_key)
    _require(isinstance(nets, dict), path,
             f"was trained with {source.algo} and carries no "
             f"{source.state_key!r} member of parameters, so it holds no actor "
             f"to export")
    memory = sorted(k for k in nets
                    if source.memory and k.startswith(source.memory))
    _require(not memory, path,
             f"holds a recurrent {source.algo} policy (memory parameters "
             f"{memory[:2]}...), and this artifact is a feedforward map with "
             f"no carried state, so exporting it would silently drop the GRU. "
             f"Recurrent export is unsupported: evaluate it on the device with "
             f"ttrl.evaluate_checkpoint, or train with recurrent unset to get "
             f"an exportable policy")
    head_key = source.heads[kind]
    _require(head_key in nets, path,
             f"holds no parameter named {head_key!r}, which is where "
             f"{source.algo}'s {kind} policy head lives")

    layers: dict[int, str] = {}
    for key in nets:
        found = source.trunk.match(key)
        if found:
            layers[int(found["index"])] = key
    _require(bool(layers), path,
             f"holds no actor trunk layers matching "
             f"{source.trunk.pattern!r}, so this file's parameter names are "
             f"not the ones {source.algo} writes")

    trunk = [_linear(nets, layers[index]) for index in sorted(layers)]
    return trunk, _linear(nets, head_key)


def _linear(nets: dict, weight_key: str) -> tuple:
    return nets[weight_key], nets[weight_key[: -len("weight")] + "bias"]
