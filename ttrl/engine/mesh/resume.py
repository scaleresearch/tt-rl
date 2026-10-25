"""Resuming a run at a different chip count than it was checkpointed at."""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["reset_optimizer_state", "resume_replicas"]


def reset_optimizer_state(optimizers: Iterable) -> None:
    """Zero every optimizer's moment buffers in place and reset its counters.

    Used on resume when the chip count changed. Optimizer moments are never
    synced across replicas, so when N changes there is no longer a
    well-defined set of moments to resume into.

    `steps`, the per-parameter update counts, is reset alongside the global
    `t`, and that is not redundant: the bias correction reads `steps[i]`.
    Zeroing the moments while leaving `steps` at, say, 50000 makes
    1 - beta**steps[i] indistinguishable from 1, so the freshly zeroed first
    moments get no bias correction and the first post-resume updates are about
    10x too small, silently, and only on this path.
    """
    # ttnn is imported here, not at module scope: ttrl.engine.mesh loads before chip
    # scoping, and ttnn must not load until the scope is set.
    import ttnn

    for opt in optimizers:
        opt.t = 0
        opt.steps = [0] * len(opt.steps)
        # Address stability: written through output_tensor, never rebound.
        for buf in list(opt.m) + list(opt.v):
            ttnn.mul(buf, 0.0, output_tensor=buf)


def resume_replicas(
    modules: Sequence,
    optimizers: Sequence,
    state: dict,
    *,
    prev_n: int,
    new_n: int,
    devices: Sequence,
) -> None:
    """Resume every replica's online network from one checkpointed parameter
    set and reset per-replica optimizer state whenever the chip count changed.

    Every replica loads the identical `state`, since parameters are the one
    thing that is shared across replicas by construction of
    `ParamSync.average`. Optimizer moments, being local and unsynced, are only
    resumable when the replica set they belonged to is unchanged: a different
    N means a different partition of experience, so replica i's old moments
    are not the same learner continued even if replica i still exists.

    `devices` is one device per entry in `modules`, in the same order.
    `load_state_dict` allocates on whichever device it is handed, and replica
    i's parameters must land on replica i's own device.
    """
    if len(devices) != len(modules):
        raise ValueError(
            f"resume_replicas: {len(devices)} devices for {len(modules)} modules; "
            "each replica loads onto its own device, so the two must correspond."
        )
    if prev_n != new_n:
        # Validated before any mutation, so raising leaves every replica
        # exactly as it was rather than half-resumed. weight_decay in
        # particular cannot be copied post-construction: Adam's weight-decay
        # scratch buffer is allocated once from whether it was nonzero at
        # construction. These should already match across replicas, built as
        # they are from one run config; if they do not, that is a real
        # misconfiguration to raise on.
        ref = optimizers[0]
        key = (ref.b1, ref.b2, ref.eps, ref.weight_decay)
        mismatched = [
            i
            for i, opt in enumerate(optimizers[1:], start=1)
            if (opt.b1, opt.b2, opt.eps, opt.weight_decay) != key
        ]
        if mismatched:
            raise ValueError(
                f"resume_replicas: optimizer hyperparameters (b1, b2, eps, "
                f"weight_decay) differ between replica 0 {key} and replica(s) "
                f"{mismatched}. Every replica must be built from the same optimizer "
                "config; resuming with mismatched hyperparameters is not supported."
            )

    for module, device in zip(modules, devices):
        module.load_state_dict(state, device)
    if prev_n != new_n:
        reset_optimizer_state(optimizers)
        # reset_optimizer_state deliberately leaves lr alone, since a
        # changed-N resume invalidates moments and not the learning rate. But
        # lr is never otherwise propagated here: replica 0's optimizer already
        # carries the checkpoint's lr, while every other replica's is freshly
        # built and still holds whatever this run's config passed in. Left
        # unfixed, replicas 1..new_n-1 train at the wrong absolute learning
        # rate from frame 0, a silent per-chip divergence that the next
        # average() smears together.
        for opt in optimizers[1:]:
            opt.lr = optimizers[0].lr
