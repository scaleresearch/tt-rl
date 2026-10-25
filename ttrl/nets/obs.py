"""Observations as the device tensors an encoder consumes.

This is the other half of the `ObsSpec` seam: `ttrl.nets.encoders` turns a spec
into a tower, and this module turns a host observation batch into the tensors
that tower takes. A caller should never need to know a tower's internal layout
to feed it, so both halves live here rather than in whichever algorithm needed
one first.

Every network in `ttrl.nets.models` is called with an `ObsPair`, the actor's
half of one observation batch and the critic's, and each reads the half it was
built for. That is what keeps privileged simulator state away from an actor: a
policy built from `ObsSpec.actor` reads `pair.actor`, and no call site chooses
between the two.

An observation is its leaves, so each half is a mapping from leaf name to one
device tensor and `ObsBuffer` holds one leaf. A single-array observation is
one leaf and one buffer, which is the buffer that was always uploaded.

The layout of each leaf is a function of its `ObsSpec.kind` alone: a vector
observation is one row per lane, an image observation is ttnn's flattened NHWC
form, which is what `ttrl.engine.conv.Conv2d` takes. `ObsSpec.scale` is not
applied here, because `ttrl.nets.encoders` folds it into the first layer's
weight.

A convolution fixes its batch at construction, so a net built for a training
minibatch cannot be called on a smaller acting batch. `ObsBuffer` binds one
fixed-size buffer, and `ttrl.nets.acting.ActingNet` keeps one buffer and one
view of the net per width a caller actually uses, so a single-lane acting step
uploads and convolves one row rather than a minibatch of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.engine.autograd import Tensor
from ttrl.engine.step.inputs import InputBinder
from ttrl.spaces import ObsSpec, require_group

__all__ = ["GroupBuffer", "ObsBuffer", "ObsPair", "ObsViews", "obs_rows",
           "obs_tensor_shape"]


def obs_tensor_shape(obs_spec: ObsSpec, batch: int) -> tuple[int, ...]:
    """Shape of the device tensor holding `batch` observations."""
    if batch < 1:
        raise ValueError(f"obs_tensor_shape: batch must be >= 1, got {batch}")
    if obs_spec.kind == "vector":
        return (int(batch), obs_spec.size)
    channels, height, width = obs_spec.shape
    return (1, 1, int(batch) * height * width, channels)


def obs_rows(obs, obs_spec: ObsSpec) -> np.ndarray:
    """`(batch, *obs_spec.shape)` host observations -> that device layout, fp32.

    An image is transposed NCHW to NHWC and flattened, which is the one host
    reordering the convolution's layout requires.
    """
    arr = np.asarray(obs)
    batch = int(arr.shape[0])
    if arr.shape[1:] != tuple(obs_spec.shape):
        raise ValueError(
            f"observation batch has element shape {arr.shape[1:]}, ObsSpec "
            f"declares {tuple(obs_spec.shape)}. Pass the observation the "
            f"environment emits, or build the network from the spec that "
            f"matches it."
        )
    if obs_spec.kind == "vector":
        return arr.reshape(batch, obs_spec.size).astype(np.float32, copy=False)
    channels, height, width = obs_spec.shape
    nhwc = arr.transpose(0, 2, 3, 1).astype(np.float32)
    return nhwc.reshape(1, 1, batch * height * width, channels)


@dataclass(frozen=True)
class ObsPair:
    """One observation batch on device, as its two readers see it.

    Each half is the leaf-name-to-tensor mapping its reader's encoder takes:
    `actor` carries the leaves and fields the actor may read and `critic`
    carries the whole observation. They are the same mapping when nothing is
    critic-only, so a symmetric run uploads one set of buffers and both
    readers see it.
    """

    actor: dict
    critic: dict

    @classmethod
    def shared(cls, tensors: dict) -> "ObsPair":
        """Both halves of a symmetric observation, which is one mapping."""
        return cls(actor=tensors, critic=tensors)


class GroupBuffer:
    """One device buffer per leaf of an `ObsGroup`, written in place.

    Address stability: each leaf's buffer is allocated once here and only ever
    written into, so `tensors` is a fixed mapping a captured trace keeps
    replaying against. Leaf order is the group's declaration order, which is
    the order a multi-modal encoder concatenates features in.
    """

    def __init__(self, obs_group, device, *, batch: int):
        self.obs_group = require_group(obs_group, "GroupBuffer")
        self.buffers = {
            name: ObsBuffer(spec, device, batch=batch)
            for name, spec in self.obs_group.specs()
        }
        self.tensors = {name: buf.tensor for name, buf in self.buffers.items()}

    def load(self, observation) -> dict:
        """Write one host observation into every leaf's buffer."""
        obs = self.obs_group.observation(observation)
        for name, buf in self.buffers.items():
            buf.load(obs[name])
        return self.tensors

    def close(self) -> None:
        for buf in self.buffers.values():
            buf.close()


class ObsViews:
    """The actor's and the critic's buffers for one observation width.

    Two sets of buffers when something is critic-only, and one otherwise: the
    actor's half is then the same leaves and the same columns as the critic's,
    and a second set would upload the same numbers twice.
    """

    def __init__(self, obs_group, device, *, batch: int):
        self.obs_group = require_group(obs_group, "ObsViews")
        self.critic = GroupBuffer(self.obs_group, device, batch=batch)
        actor_group = self.obs_group.actor
        self.actor = (self.critic if actor_group is self.obs_group
                      else GroupBuffer(actor_group, device, batch=batch))
        #: Address stability: the buffers are never rebound, so this one pair
        #: of mappings is what every load writes into and what a captured trace
        #: keeps replaying against.
        self.pair = ObsPair(actor=self.actor.tensors, critic=self.critic.tensors)

    def load(self, obs) -> ObsPair:
        """Write one host observation batch into both halves."""
        whole = self.obs_group.observation(obs)
        self.critic.load(whole)
        if self.actor is not self.critic:
            self.actor.load(self.obs_group.actor_observation(whole))
        return self.pair

    def close(self) -> None:
        """Release both halves' staging. Idempotent."""
        self.critic.close()
        self.actor.close()


class ObsBuffer:
    """One fixed-shape device buffer for observations, written in place.

    Address stability: the buffer is allocated once and only ever written
    into, never rebound, so a captured trace keeps replaying against the
    address it recorded.

    The upload goes through `InputBinder`, the same one `TrainStep` binds its
    inputs with, so the device-tilize staging it owns covers acting as well as
    learning. An observation buffer is the largest thing either path uploads
    and the one that staging was measured on.
    """

    def __init__(self, obs_spec: ObsSpec, device, *, batch: int):
        self.obs_spec = obs_spec
        self.batch = int(batch)
        self.shape = obs_tensor_shape(obs_spec, self.batch)
        self._rows = np.zeros(self.shape, dtype=np.float32)
        self.tensor = Tensor.from_numpy(self._rows, device, requires_grad=False)
        self._binder = InputBinder(device, {"obs": self.tensor})

    def rows(self, obs) -> np.ndarray:
        """Host rows for `obs`, padded up to this buffer's fixed batch.

        A batch shorter than the buffer leaves the tail at whatever it last
        held. That is deliberate: the caller slices its result back to the
        real batch, and zeroing the tail would cost a pass over the buffer on
        every acting step to change numbers nobody reads.
        """
        arr = np.asarray(obs)
        n = int(arr.shape[0])
        if n > self.batch:
            raise ValueError(
                f"ObsBuffer holds {self.batch} observations but was given {n}. "
                f"The network's convolutions fix their batch at construction, "
                f"so build the agent with a batch of at least {n}."
            )
        rows = obs_rows(arr, self.obs_spec)
        if n == self.batch:
            return rows
        if self.obs_spec.kind == "vector":
            self._rows[:n] = rows
        else:
            self._rows[0, 0, : rows.shape[2]] = rows[0, 0]
        return self._rows

    def load(self, obs) -> Tensor:
        """Write `obs` into the bound buffer and return it."""
        self._binder.load("obs", self.rows(obs))
        return self.tensor

    def close(self) -> None:
        """Release the staging buffer. Idempotent."""
        self._binder.close()
