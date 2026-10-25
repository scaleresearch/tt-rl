"""Opening one MeshDevice spanning several chips, by discovery.

The shape is never supplied and no descriptor is forced. ttnn discovers the
physical mesh unaided, logs "Using auto discovery to generate mesh graph" and
hands back the shape the cabling actually has, which is the only shape that
can be realised. It owns a table of its own over roughly fourteen cluster
types, so any table here could only be a worse copy of it, and ours was also
wrong: it asserted (1, 2) for a P300 whose hardware reports [2, 1].

`open_mesh_device` needs a shape only when `physical_device_ids` is supplied
("Mesh shape must be provided when physical device ids are supplied"), so
scoping the process and passing neither is what makes discovery the only
source of the shape.

There is no public API that enumerates the legal placements: the enumerator
lives in an anonymous namespace, and what is exposed is the one discovered
shape plus a size assertion. So a wanted placement cannot be checked in
advance, only attempted, which is why the validation below runs after the open
and says what to do about it.
"""

from __future__ import annotations

import contextlib
from typing import Sequence

from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.device import purge_device_caches
from ttrl.engine.mesh.topology import (
    ParamSyncError,
    local_ids_for,
    require_exact_visible_scope,
)

__all__ = ["chips_of_open_mesh", "open_mesh_for_chips", "replica_axis"]

#: The chips of the one mesh open in this process, or None. Module state
#: because the invariant it enforces is per-process: see `_claim`.
_OPEN_MESH_CHIPS: list[int] | None = None


def chips_of_open_mesh() -> list[int] | None:
    """The chips of this process's open mesh, or None if none is open."""
    return None if _OPEN_MESH_CHIPS is None else list(_OPEN_MESH_CHIPS)


def _claim(chips: Sequence[int]) -> None:
    """Refuse a second mesh in this process, naming the first.

    Not advisory. Two disjoint meshes opened in one process, with one then
    closed, leave the other unusable: `synchronize_device` never returns,
    spinning on a core with the stack in `FDMeshCommandQueue::wait_for_outstanding_reads`.
    That is a permanent hang with no error, so it has to be refused here,
    before any device is touched. The runtime's own recommendation is
    `create_unit_meshes()`, which ttnn 0.80 does not bind in Python.
    """
    global _OPEN_MESH_CHIPS
    if _OPEN_MESH_CHIPS is not None:
        raise ParamSyncError(
            f"this process already holds a mesh over chips {_OPEN_MESH_CHIPS}, so it "
            f"cannot open a second one over {list(chips)}. One MeshDevice per process "
            "is a hard limit of the runtime: a second mesh makes the first one hang "
            "forever in synchronize_device, with no error. Close the first mesh by "
            "leaving its open_mesh_for_chips block, or run the second mesh in its own "
            "process scoped with scope_to_chips()."
        )
    _OPEN_MESH_CHIPS = [int(c) for c in chips]


def _release() -> None:
    global _OPEN_MESH_CHIPS
    _OPEN_MESH_CHIPS = None


def replica_axis(mesh) -> int:
    """The axis of `mesh.shape` that N replicas are laid out along.

    Read off the opened mesh and never assumed. Axis 0 is not the replica axis
    on this box: a P300 pair auto-discovers as MeshShape([2, 1]), while the
    installed p300 descriptor says device_topology dims [1, 2], so a hardcoded
    axis is wrong under one of the two.
    """
    dims = [int(d) for d in mesh.shape]
    total = 1
    for d in dims:
        total *= d
    if total == 1:
        # Every axis of a one-device mesh has extent 1, so the test below is
        # ambiguous. There is no cross-chip reduction to place anyway.
        return 0
    axes = [i for i, d in enumerate(dims) if d == total]
    if len(axes) != 1:
        raise ParamSyncError(
            f"mesh shape {dims} is not a line, so no single axis carries all "
            f"{total} replicas. Data-parallel replicas need a 1xN or Nx1 mesh; open "
            "a submesh of that shape, or pass cluster_axis explicitly to the "
            "collective."
        )
    return axes[0]


@contextlib.contextmanager
def open_mesh_for_chips(chips: Sequence[int], *, cfg: EngineConfig = DEFAULT):
    """Open one MeshDevice covering `chips` by discovery and yield it.

    `chips` are physical ids. The process must already be scoped to exactly
    them, which is checked before ttnn is imported, because that is the last
    moment at which a wrong scope can still be fixed.
    """
    chips = [int(c) for c in chips]
    require_exact_visible_scope(chips)
    local_ids = local_ids_for(chips)
    _claim(chips)

    # ttnn is imported here, not at module scope: ttrl.engine.mesh loads before chip
    # scoping, and ttnn must not load until the scope is set.
    import ttnn

    # Deferred: ttrl.engine.mesh.devices imports this module.
    from ttrl.engine.mesh.devices import lock_chips

    try:
        with contextlib.ExitStack() as stack:
            lock_chips(stack, local_ids)
            # Before the open, never after: changing the fabric config while
            # devices are open is refused with "SetFabricConfig(FABRIC_1D) is
            # not allowed while devices are still open".
            #
            # FABRIC_1D is a precondition of the collective and a hard error
            # below two chips: "Fabric config FABRIC_1D requires at least 2
            # participating chips, but the opened mesh has 1 local device(s)",
            # fatal at open time. Measured on ttnn 0.80.0.
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D
                if len(chips) > 1
                else ttnn.FabricConfig.DISABLED
            )
            try:
                mesh = ttnn.open_mesh_device(
                    l1_small_size=cfg.l1_small_size,
                    trace_region_size=cfg.trace_region_size,
                )
            except Exception as e:
                raise ParamSyncError(
                    f"ttnn could not open a mesh over chips {chips}: {e}. ttnn "
                    "discovers the shape itself and there is no API that lists the "
                    "legal placements, so this set of chips has no mesh ttnn will "
                    "accept. Ask for a whole board (both chips of one P300) rather "
                    "than halves of two, and check that TT_MESH_GRAPH_DESC_PATH is "
                    "unset so discovery runs."
                ) from e
            try:
                _check_devices(mesh, chips)
                mesh.enable_program_cache()
                yield mesh
            finally:
                _close(mesh)
    finally:
        _release()


def _check_devices(mesh, chips: Sequence[int]) -> None:
    """Validate after the open, since nothing can be asked in advance."""
    ids = list(mesh.get_device_ids())
    if len(ids) != len(chips):
        raise ParamSyncError(
            f"asked for {len(chips)} chips {list(chips)} but the discovered mesh "
            f"{list(mesh.shape)} holds {len(ids)} devices {ids}. Discovery reports the "
            "cabling, so the visible set and the wanted chips disagree: relaunch "
            f"scoped with scope_to_chips({list(chips)}) before ttnn is imported."
        )


def _close(mesh) -> None:
    # ttnn is imported here, not at module scope: ttrl.engine.mesh loads before chip
    # scoping, and ttnn must not load until the scope is set.
    import ttnn

    # Purge inside the try so a failing purge still closes the mesh, and drop
    # the fabric config only once it is closed, for the ordering above.
    try:
        purge_device_caches(mesh)
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
