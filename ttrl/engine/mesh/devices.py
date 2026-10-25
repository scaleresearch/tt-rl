"""Opening N genuinely independent single-chip devices in one process."""

from __future__ import annotations

import contextlib
from typing import Sequence

from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.device import purge_device_caches, status
from ttrl.engine.device.liveness import hold_lock
from ttrl.engine.mesh.open import chips_of_open_mesh
from ttrl.engine.mesh.topology import (
    ParamSyncError,
    local_ids_for,
    require_exact_visible_scope,
)

__all__ = ["open_independent_devices_for_chips"]


def lock_chips(stack: contextlib.ExitStack, local_ids: Sequence[int]) -> None:
    """Liveness-check and advisory-lock every chip, all or nothing.

    Calls straight into the device module's primitives rather than
    duplicating `acquire`'s cooperative checks. The locks go onto `stack`, so
    if any chip is busy the locks already taken are released before the
    exception leaves this call.
    """

    for local_id in local_ids:
        st = status(local_id)
        if not st.free:
            raise ParamSyncError(
                f"chip {st.path} is not free (foreign_pids={st.foreign_pids}, "
                f"locked_by={st.locked_by}). Refusing to open a device onto a chip "
                "another process is using."
            )
        if not stack.enter_context(hold_lock(local_id)):
            raise ParamSyncError(
                f"could not acquire the advisory lock for chip {st.path}: another ttrl "
                "process took it between the liveness check and now."
            )


@contextlib.contextmanager
def open_independent_devices_for_chips(
    chips: Sequence[int], *, cfg: EngineConfig = DEFAULT
):
    """Open one independent single-chip device per entry in `chips` and yield
    them in the same order.

    Never a shared MeshDevice. See `ttrl.engine.mesh.sync` for why a mesh
    cannot coexist with these.

    Sequential `ttnn.open_device` per chip deadlocks on real hardware once a
    parameter is downloaded; the batch CreateDevices/CloseDevices API used
    here does not.
    """
    chips = list(chips)
    held = chips_of_open_mesh()
    if held is not None:
        raise ParamSyncError(
            f"this process holds a mesh over chips {held}, so it cannot also open "
            f"independent devices on {chips}. MeshDeviceImpl::ScopedDevices adopts "
            "every device active in the context and closes all of them, with no "
            "refcount, so the two cannot coexist: opening a standalone device onto a "
            "chip under an open fabric-configured mesh times out waiting for an "
            "ethernet core to retrain and leaves that chip's link degraded until "
            "tt-smi -r. Leave the open_mesh_for_chips block first."
        )
    # Both checks run before ttnn is imported here deliberately: the scope
    # check's whole point is a mistake that is only fixable before the first
    # import of ttnn, so it must not be reported by a process that has already
    # taken the driver locks it is complaining about.
    require_exact_visible_scope(chips)
    local_ids = local_ids_for(chips)

    # ttnn is imported here, not at module scope: ttrl.engine.mesh loads before chip
    # scoping, and ttnn must not load until the scope is set.
    import ttnn

    with contextlib.ExitStack() as lock_stack:
        lock_chips(lock_stack, local_ids)

        devices_by_id = ttnn.CreateDevices(
            local_ids,
            l1_small_size=cfg.l1_small_size,
            trace_region_size=cfg.trace_region_size,
        )
        try:
            missing = [i for i in local_ids if i not in devices_by_id]
            if missing:
                raise ParamSyncError(
                    f"ttnn.CreateDevices({local_ids}) returned no device for {missing} "
                    f"(got {sorted(devices_by_id)})."
                )
            devices = [devices_by_id[i] for i in local_ids]
            for dev in devices:
                dev.enable_program_cache()
            yield devices
        finally:
            # Inside the try, so a failed lookup still closes every chip
            # CreateDevices opened rather than leaking all N until exit.
            #
            # Purge first and per device. N devices live side by side here, so
            # every id(device)-keyed cache holds N sets of entries, and the
            # moment these are freed those ids go to the next opener.

            # Nested: one purge raising partway through the loop would
            # otherwise skip CloseDevices and leak every chip this opener
            # took, not just the one whose purge failed.
            try:
                for dev in devices_by_id.values():
                    purge_device_caches(dev)
            finally:
                ttnn.CloseDevices(devices_by_id)
