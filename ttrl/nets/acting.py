"""Calling a network at whatever batch width arrives.

A convolution is the only module whose shapes depend on the batch it was built
for, so a net built for a training minibatch needs a narrower view of itself to
run an acting step. `net_at_batch` builds that view against the same master
weights and `ActingNet` keeps one per width a caller actually uses.

`check_encoder_layout` lives here for the same reason: it is about what the
built net will accept, which is the question this module answers.
"""

from __future__ import annotations

import copy

import numpy as np

from ttrl.engine.conv import Conv2d
from ttrl.engine.nn import Module, ModuleList
from ttrl.nets.obs import ObsViews
from ttrl.spaces import ObsGroup, require_group

__all__ = ["ActingNet", "check_encoder_layout", "net_at_batch"]


def check_encoder_layout(net) -> None:
    """Refuse a net whose convolutions want an input this module does not build.

    A `fold > 1` stage (the space-to-depth spelling of a convolution) consumes
    activations whose spatial extent is divided and whose channels are
    multiplied, which is a different host reordering from the plain flattened
    NHWC `ttrl.nets.obs.obs_rows` produces. Feeding the plain layout to a
    folded conv convolves permuted pixels and still trains, so it is refused
    here.
    """
    for name, module in _walk_modules(net):
        geom = getattr(module, "geom", None)
        if geom is not None and int(getattr(geom, "fold", 1)) != 1:
            raise ValueError(
                f"{name} is a convolution with fold={geom.fold} (space-to-depth). "
                f"ttrl.nets.obs builds plain flattened-NHWC activations, so "
                f"this layer would convolve reordered pixels. Build the agent "
                f"with encoder='nature_cnn' instead of a folded tower."
            )


def _walk_modules(module, prefix: str = "net", seen: set[int] | None = None):
    """Every `Module` in a tree, named by the path that reaches it.

    Walks `_slot_sources`, the single hook `ttrl.engine.nn` exposes for a
    container's children, so a `ModuleList` or a `ModuleDict` is traversed by
    the same rule as an ordinary attribute and nothing is skipped.
    """

    seen = set() if seen is None else seen
    if not isinstance(module, Module) or id(module) in seen:
        return
    seen.add(id(module))
    yield prefix, module
    for name, child in module._slot_sources():
        children = (child.items() if isinstance(child, dict)
                    else enumerate(child) if isinstance(child, (list, tuple))
                    else [(None, child)])
        for key, item in children:
            path = f"{prefix}.{name}" if key is None else f"{prefix}.{name}.{key}"
            yield from _walk_modules(item, path, seen)


def net_at_batch(net, batch: int, _seen: dict | None = None):
    """A view of `net` taking `batch` observations, sharing its parameters.

    A convolution is the only module whose shapes depend on the batch it was
    built for: a matmul, a flatten and a head each read the row count off
    their input. So the convolutions are rebuilt at `batch` against the same
    master weights and everything else is reused, which leaves one set of
    parameters for the optimizer to update and for every view to see.

    ONE SOURCE MODULE BECOMES ONE VIEW MODULE. Two slots holding the same
    object in the net hold the same object in the view, which is what
    `_seen` keeps. Architectures here say things with identity: a symmetric
    `ObsSpec` puts ONE trunk in both of `ActorCritic`'s slots and its
    `forward` runs it once because `critic is actor`, and
    `ttrl.nets.recurrent.RecurrentActorCritic` carries one memory rather than
    two by the same test. Copying each slot independently silently turns a
    shared trunk into two trunks that run twice and a shared memory into two
    hidden states, at acting width only, which is the width no gradcheck
    covers.
    """

    seen = {} if _seen is None else _seen
    key = id(net)
    if key in seen:
        return seen[key]

    if isinstance(net, Conv2d):
        view = net.at_batch(batch)
    elif isinstance(net, ModuleList):
        view = ModuleList([net_at_batch(child, batch, seen) for child in net])
    elif isinstance(net, dict):
        view = {k: net_at_batch(v, batch, seen) for k, v in net.items()}
    elif isinstance(net, (list, tuple)):
        view = type(net)(net_at_batch(child, batch, seen) for child in net)
    elif not isinstance(net, Module):
        return net
    else:
        view = copy.copy(net)
        # Recorded before the children are walked, so a module reachable from
        # its own subtree resolves to this view rather than recursing.
        seen[key] = view
        for name, value in vars(net).items():
            setattr(view, name, net_at_batch(value, batch, seen))
        return view
    seen[key] = view
    return view


class ActingNet:
    """A net called at whatever lane count arrives, one buffer pair per width.

    `width` is the batch the net was built for, and the widest call this
    accepts. Narrower widths get their own view and their own buffers, built on
    first sight and then kept: building probes a convolution's layout on
    device, which is far too slow to repeat per step.

    Evaluation runs a different number of lanes from training and decides how
    many from its own `EnvConfig`, so the set of widths is not knowable at
    construction. Keying on the width that arrives is what keeps a narrow
    acting step narrow without making the eval width a second config field
    that has to agree with the first.

    The host observations passed in are whole observations as the environment
    emits them, which is an `Observation` over the group's leaves. Cutting
    them down to what the actor may read is `ObsViews`' job, so no caller of
    this class handles the privileged half.
    """

    def __init__(self, net, obs_spec: ObsGroup, device, *, width: int, lanes: int):
        self.obs_spec = require_group(obs_spec, "ActingNet")
        self.device = device
        self.width = int(width)
        self._net = net
        self._views = {self.width: (net, ObsViews(obs_spec, device, batch=self.width))}
        self._view(int(lanes))

    def _view(self, lanes: int):
        if lanes > self.width:
            raise ValueError(
                f"ActingNet was built for at most {self.width} observations and "
                f"was given {lanes}. A convolution fixes its batch at "
                f"construction, so raise the network batch to at least {lanes} "
                f"or run fewer environments."
            )
        view = self._views.get(lanes)
        if view is None:
            view = self._views[lanes] = (
                net_at_batch(self._net, lanes),
                ObsViews(self.obs_spec, self.device, batch=lanes),
            )
        return view

    def __call__(self, obs, *args, **kwargs):
        observation = self.obs_spec.observation(obs)
        net, views = self._view(int(observation.lanes))
        return net(views.load(observation), *args, **kwargs)

    def step(self, obs, *args, **kwargs):
        """The net's single-timestep entry point at this width.

        A recurrent network has two entry points, an unroll over a sequence
        and one step from a caller-owned carry, and only the caller knows
        which it is asking for. A feedforward network has no `step` and a call
        here says so by name.
        """
        observation = self.obs_spec.observation(obs)
        net, views = self._view(int(observation.lanes))
        step = getattr(net, "step", None)
        if step is None:
            raise TypeError(
                f"{type(net).__name__} has no single-timestep `step`; only a "
                f"recurrent network carries state from one acting step to the "
                f"next"
            )
        return step(views.load(observation), *args, **kwargs)

    def close(self) -> None:
        """Release every width's staging buffers. Idempotent."""
        for _net, views in self._views.values():
            views.close()
