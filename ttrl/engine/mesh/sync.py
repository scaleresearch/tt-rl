"""ParamSync: periodic exact parameter averaging across N replicas.

WHY THE REDUCTION IS ON THE HOST, which is the only record of why the obvious
design is wrong. The reason is NOT that the device collective is broken; it
has simply never been run on this version. The "shards carrying 1 and 2 both
come back 3" result recorded here was measured on an older ttnn, against an
`all_reduce` that took `math_op` and `mesh_device`; 0.80.0 keyword-accepts only
`cluster_axis`, `subdevice_id`, `memory_config`, `num_links` and `topology`,
so that call no longer type-checks and the figure cannot be claimed for 0.80.
No shipped path calls it. The old ethernet-core timeout was never about the
collective at all; it was about opening a standalone device onto a chip
already under an open fabric-configured mesh, which is the coexistence problem
below and is still true.

The blocker is the shape of the replica set. `ttnn.all_reduce` consumes one
multi-device tensor living on a single MeshDevice. Each replica here holds its
own independently-opened single-chip device, by construction, so its
parameters are N unrelated single-device tensors and there is no multi-device
tensor to reduce. There cannot safely be one either: a MeshDevice and
independently-opened devices on the same chips cannot coexist in one process,
because MeshDeviceImpl::ScopedDevices adopts ALL devices active in the context
rather than the ones it asked for, and its destructor closes every one of
them, with no refcount anywhere. Measured consequences: opening a standalone
device for a chip already under an open fabric-configured mesh times out
waiting for an ethernet core to retrain, SIGSEGVs on mesh teardown, and leaves
that chip's link degraded until a hard `tt-smi -r`; opening the mesh while
standalone devices are open raises "SetFabricConfig(FABRIC_1D) is not allowed
while devices are still open".

So using the device collective is not a change to this file. It means running
every replica as a shard of one MeshDevice, which the engine's tensor,
optimizer and trace layers do not do, and `ttrl.engine.mesh.open` is the piece
of that groundwork which exists. Until then the host round trip stays, and it
is a per-sync device-to-host-to-device copy of the whole parameter set.

The reduction is therefore numpy on the host. The contract is unchanged: exact
fp32 sum here, divided by N by `_all_reduce_param`.

KNOWN LIMITATION, no timeout on the download path: a hung chip, as opposed to
one that crashes, blocks `average` forever. `ttnn.Tensor.cpu()` takes only
`blocking` and `cq_id`, with no deadline, and there is no cancellation point
inside the C++ extension for a userspace watchdog to unwind safely. A real fix
needs an upstream timeout primitive or a supervising process that can kill the
replica and reset the chip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ttrl.engine.mesh.topology import ParamSyncError
from ttrl.engine.tensor import restore_in_place

__all__ = ["ParamSync", "host_all_reduce_sum", "param_schema", "replica_schema"]


def host_all_reduce_sum(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Exact fp32 elementwise sum, identical on every returned entry.

    Callers divide by len(arrays) themselves, so the contract stays
    exact-sum-then-divide rather than whatever an averaging collective happens
    to do internally.
    """
    summed = np.asarray(arrays[0], dtype=np.float32).copy()
    for a in arrays[1:]:
        summed = summed + np.asarray(a, dtype=np.float32)
    return [summed for _ in arrays]


def param_schema(module) -> tuple[tuple[str, tuple], ...]:
    """((name, shape), ...) for one replica's parameters, in order.

    Comparing names and not only shapes is what catches two same-shaped
    parameters swapped between replicas; averaging those mixes unrelated
    tensors and diverges the run with no error anywhere.
    """
    params = list(module.parameters())
    names = [n for n, _ in module.named_parameters()]
    if len(names) != len(params):
        raise ParamSyncError(
            f"{type(module).__name__}: {len(names)} parameter names for "
            f"{len(params)} parameters; named_parameters() and parameters() disagree."
        )
    return tuple((str(n), tuple(p.shape)) for n, p in zip(names, params))


def replica_schema(modules: Sequence) -> tuple[tuple[str, tuple], ...]:
    """The one parameter schema every replica must share, or raise."""
    schemas = [param_schema(m) for m in modules]
    if len(set(schemas)) != 1:
        raise ParamSyncError(
            "replica parameter schemas differ by name or shape, so a positional "
            f"average would mix unrelated tensors: {schemas}. Every replica must build "
            "the identical model in the identical order."
        )
    return schemas[0]


@dataclass
class ParamSync:
    """Periodic exact parameter averaging across N replicas.

    `modules` is one module per replica, in replica order, which is the order
    every call site must agree on since `average` zips them positionally.
    """

    modules: list
    sync_every: int = 1
    reduce_fn: Callable[[list[np.ndarray]], list[np.ndarray]] = field(
        default=host_all_reduce_sum, repr=False
    )
    _updates_since_sync: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sync_every < 1:
            raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
        self.n = len(self.modules)
        if self.n < 1:
            raise ValueError("ParamSync needs at least one replica, got none")
        # Parameter order and identity must match across replicas, because
        # everything below zips per-replica parameter lists positionally.
        self._schema = replica_schema(self.modules)
        self._nparams = len(self._schema)

    def maybe_average(self, targets: Sequence | None = None) -> bool:
        """Call once per learner update. Averages every `sync_every` calls and
        returns whether it did."""
        self._updates_since_sync += 1
        if self._updates_since_sync >= self.sync_every:
            self._updates_since_sync = 0
            self.average(targets=targets)
            return True
        return False

    def average(self, targets: Sequence | None = None) -> None:
        """All-reduce every parameter across replicas in place, fp32,
        exact sum then divide, then refresh each `targets` entry from its own
        now-averaged online net.

        Optimizer state is never touched: only `parameters()` is read and
        written, so each replica's moments necessarily stay local.
        """
        # Validated before any parameter is touched, so a wrong-length
        # `targets` cannot leave the online nets averaged while raising.
        if targets is not None and len(targets) != self.n:
            raise ValueError(f"expected {self.n} target modules, got {len(targets)}")
        self._require_schema("average")

        per_replica = [list(m.parameters()) for m in self.modules]
        # Not atomic: each parameter is reduced and written back on its own, so
        # a collective that raises partway leaves earlier parameters averaged
        # and later ones not. An atomic rewrite needs a second scratch buffer
        # per parameter, doubling device memory for the parameter set. Instead
        # the error below names what is and is not averaged, so a caller never
        # mistakes this for a clean failure.
        for j in range(self._nparams):
            shards = [per_replica[i][j] for i in range(self.n)]
            try:
                self._all_reduce_param(shards)
            except Exception as e:
                raise ParamSyncError(
                    f"ParamSync.average(): the reduction failed on parameter "
                    f"{j}/{self._nparams}: {e}. Parameters 0..{j - 1} may already be "
                    f"fully averaged across replicas; parameter {j} is written to each "
                    "replica sequentially inside its own reduction, so it may now be "
                    "PARTIALLY propagated, with some replicas holding the averaged "
                    f"value and others their own pre-call value; parameters {j + 1}.."
                    f"{self._nparams - 1} are untouched. The replicas are in a "
                    "partially-averaged, inconsistent state. Do not resume training or "
                    "checkpoint from this run without re-syncing (for example "
                    "broadcast() from a known-good replica) first."
                ) from e

        self._refresh_targets(targets)

    def broadcast(self, src: int = 0, targets: Sequence | None = None) -> None:
        """Overwrite every replica's online parameters with replica `src`'s.

        This, and not `average`, is how "N replicas of one model" is made true
        before frame 0. Averaging N independently seeded initialisations is a
        different distribution: each averaged weight is the mean of n i.i.d.
        draws, so its standard deviation is the configured init's divided by
        sqrt(n), and a 4-chip run would start from a network at half the
        intended weight scale, with the starting model silently depending on
        the chip count.

        Only online parameters move. Per-replica env, action and replay RNG
        streams stay distinct, which is the divergence that makes N
        actor-learners worth having.
        """
        if not 0 <= src < self.n:
            raise ValueError(f"src replica {src} out of range for {self.n} replicas")
        if targets is not None and len(targets) != self.n:
            raise ValueError(f"expected {self.n} target modules, got {len(targets)}")
        self._require_schema("broadcast")

        per_replica = [list(m.parameters()) for m in self.modules]
        for j in range(self._nparams):
            source = np.asarray(per_replica[src][j].numpy(), dtype=np.float32)
            for i in range(self.n):
                if i != src:
                    restore_in_place(per_replica[i][j].data, source)

        self._refresh_targets(targets)

    def _require_schema(self, what: str) -> None:
        # Revalidated every call, not only at construction: a replica whose
        # parameter list grew, shrank or reordered since then would otherwise
        # be detected partway through the write loop.
        current = replica_schema(self.modules)
        if current != self._schema:
            raise ParamSyncError(
                "the replica parameter schema changed since ParamSync was built: "
                f"{self._schema} -> {current}. Refusing to {what}."
            )

    def _refresh_targets(self, targets: Sequence | None) -> None:
        # After the online buffers are written, so a target sync coinciding
        # with a param sync always refreshes from the averaged weights.
        if targets is None:
            return
        for online, target in zip(self.modules, targets):
            target.copy_from(online)

    def _all_reduce_param(self, shards) -> None:
        if self.n == 1:
            return
        arrays = [np.asarray(s.numpy(), dtype=np.float32) for s in shards]
        shape = arrays[0].shape
        for a in arrays[1:]:
            if a.shape != shape:
                raise ParamSyncError(
                    f"parameter shape mismatch across replicas: {shape} vs {a.shape}"
                )
        summed = self.reduce_fn(arrays)
        if len(summed) != len(shards):
            raise ParamSyncError(
                f"reduce_fn returned {len(summed)} results for {len(shards)} replica "
                "shards; a short result would silently leave some replicas unaveraged."
            )
        n = np.float32(self.n)
        for shard, total in zip(shards, summed):
            # Address stability: restore_in_place writes through the existing
            # buffer and never rebinds it, so a sync leaves captured traces
            # valid exactly as a checkpoint restore does.
            restore_in_place(shard.data, np.asarray(total, dtype=np.float32) / n)
