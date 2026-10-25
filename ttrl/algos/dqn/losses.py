"""DQN's update body, and the input buffers it reads.

Separated from the agent for the reason SAC's losses are: this is the
algorithm and the agent is bookkeeping. The agent owns the replay buffer, the
optimizer, the epsilon schedule and the target-sync clock; this owns the
arithmetic of one update and the shape of the batch that arithmetic reads.

The two halves ship together because they are one contract. `step_inputs`
names every key `update_loss` may read, and a key added to one without the
other is a KeyError on the first gradient step.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.grads import grad_sq_device
from ttrl.algos.mask import mask_penalty
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.ops import (abs_, add, apply_done_mask, argmax_dim_detached,
                             gather_onehot, huber, mean_all, mul,
                             one_hot_gather, sub)
from ttrl.nets.obs import ObsPair, obs_tensor_shape
from ttrl.spaces import ObsGroup

__all__ = ["leaf_key", "leaf_tensors", "step_inputs", "update_loss"]


def leaf_key(which: str, leaf: str) -> str:
    """The input name holding one leaf of `obs` or of `next_obs`.

    An observation is its leaves, so it occupies one named buffer per leaf
    rather than one buffer. A single-array observation is one leaf and so one
    buffer, the buffer that was always bound.
    """
    return f"{which}.{leaf}"


def leaf_tensors(inputs: dict, which: str, leaves) -> dict:
    """The leaf-to-tensor mapping an encoder takes, out of the bound inputs."""
    return {leaf: inputs[leaf_key(which, leaf)] for leaf in leaves}


def step_inputs(obs_spec: ObsGroup, device, *, batch: int, num_actions: int,
                mask_n: int, prioritized: bool = False) -> dict[str, Tensor]:
    """The fixed device buffers one update reads, allocated once.

    Address stability: every buffer here is allocated once and thereafter only
    written into, never rebound, which is what lets a captured trace keep
    seeing new batches instead of replaying the one it captured.
    """
    def zeros(*shape):
        return Tensor.from_numpy(np.zeros(shape, np.float32), device)

    inputs = {
        "rewards": zeros(batch, 1),
        "dones": zeros(batch, 1),
        "discounts": zeros(batch, 1),
        "actions_onehot": zeros(batch, num_actions),
    }
    for leaf, spec in obs_spec.specs():
        for which in ("obs", "next_obs"):
            inputs[leaf_key(which, leaf)] = zeros(*obs_tensor_shape(spec, batch))
    if mask_n:
        inputs["next_mask"] = Tensor.from_numpy(
            np.ones((batch, mask_n), np.float32), device)
    if prioritized:
        # Allocated only for a prioritized run, so a uniform one's captured
        # trace holds no multiply by a buffer of ones. Ones rather than zeros
        # at allocation: the capture's warm-up pass runs this loss for real,
        # and a zero weight there would capture a zero gradient.
        inputs["is_weights"] = Tensor.from_numpy(
            np.ones((batch, 1), np.float32), device)
    return inputs


def update_loss(online, target, inputs: dict, *, leaves, num_actions: int,
                mask_n: int, double: bool, huber_delta: float,
                prioritized: bool = False) -> dict[str, Tensor]:
    """One DQN update: the Huber loss against the n-step double-Q target.

    Reads only `inputs`, never host state, and never reads a tensor back,
    which is what makes it capturable. `backward` is called here so the whole
    forward and backward land inside one capture.

    The TD target uses the per-sample `discounts` the replay buffer carries,
    which is gamma raised to the number of rewards actually summed. A fixed
    gamma ** n_step is wrong for every window an episode boundary cut short,
    and the error is silent: those samples bootstrap with too much discount
    and the value function learns a slightly wrong horizon.
    """
    # DQN's one network both acts and is trained, so `build_q_net` refuses a
    # spec with anything critic-only and both halves here are the same mapping.
    obs = ObsPair.shared(leaf_tensors(inputs, "obs", leaves))
    next_obs = ObsPair.shared(leaf_tensors(inputs, "next_obs", leaves))
    pred = gather_onehot(online(obs), inputs["actions_onehot"])

    # Double DQN: the action is SELECTED by the online net and VALUED by the
    # target net. Both forwards here feed only a detached index or a detached
    # value, so neither needs a tape.
    with no_grad():
        next_target_q = target(next_obs)
        selector = online(next_obs) if double else next_target_q
        if mask_n:
            # The whole point of (b): the argmax runs over the successor's
            # legal set. The gather needs no penalty, because the index it
            # gathers is already legal.
            selector = add(selector, mask_penalty(inputs["next_mask"]))
        next_actions = argmax_dim_detached(selector, dim=-1)
        next_value = one_hot_gather(next_target_q, next_actions, num_actions)
        bootstrap = mul(apply_done_mask(next_value, inputs["dones"]),
                        inputs["discounts"])
        td_target = add(inputs["rewards"], bootstrap)

    # WHERE THE TRACE BOUNDARY IS, for prioritized replay. Both halves of PER
    # that touch the device sit INSIDE the captured region: the per-sample
    # weights arrive in a fixed input buffer that `load_input` fills before
    # the replay, and the per-sample TD error leaves as a metric tensor read
    # back AFTER it, like every other metric. Neither is a host read inside
    # the capture, which is what trace capture forbids. The host half, writing
    # those errors back into the priority table, is in the agent.
    errors = huber(pred, td_target, delta=huber_delta)
    if prioritized:
        # Per-sample, before the reduction: prioritized sampling is a biased
        # estimator of the uniform expectation and these weights are what make
        # the gradient estimate the same quantity. A mean taken first and
        # scaled after would weight the whole batch by one number.
        errors = mul(errors, inputs["is_weights"])
    loss = mean_all(errors)
    loss.backward()

    metrics = {"loss": loss, "mean_q": mean_all(pred)}
    if prioritized:
        with no_grad():
            metrics["td_abs"] = abs_(sub(pred.detach(), td_target))
    # Summed on device: a host read here is fatal to trace capture.
    grad_sq = grad_sq_device(online.parameters())
    if grad_sq is not None:
        metrics["grad_sq"] = Tensor(grad_sq, requires_grad=False)
    return metrics
