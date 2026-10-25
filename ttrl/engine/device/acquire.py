"""Opening a chip, and the device-keyed cache purge that must precede closing it."""

from __future__ import annotations

import contextlib
import time
import traceback
from typing import Iterator

from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.device.liveness import (
    DEFAULT_WAIT_SECONDS,
    POLL_SECONDS,
    STALE_FD_NOTE,
    NoDeviceAvailable,
    describe_holders,
    enumerate_chips,
    hold_lock,
    status,
    survey,
)
from ttrl.engine.device.profile import DeviceProfile, probe_profile
from ttrl.engine.device.scope import as_local_index

_EXTRA_PURGERS: list = []


def register_device_cache(purge) -> None:
    """Register a purge(device) for a cache keyed by id(device).

    Any module outside engine/ops that memoises under an id(device) key must
    call this at import time. See purge_device_caches for the hazard.
    """
    _EXTRA_PURGERS.append(purge)


def purge_device_caches(device) -> None:
    """Evict every process-global cache entry keyed by id(device).

    Must run before ttnn.close_device, because id() is unique only among live
    objects: once the device object is freed, the next open_device in the
    process can be handed the same address and inherit the dead device's
    caches, including caches of device-resident tensors. That is silent wrong
    arithmetic whose trigger is which device was opened when.
    """
    # Deferred: ttrl.engine.ops imports ttnn, and this module loads before chip scoping.
    from ttrl.engine.ops import purge_lane_caches, purge_matmul_caches

    purge_matmul_caches(device)
    purge_lane_caches(device)
    for purge in _EXTRA_PURGERS:
        purge(device)


@contextlib.contextmanager
def acquire(
    chip_id: int | None = None,
    *,
    cfg: EngineConfig = DEFAULT,
    wait: bool = True,
    timeout: float | None = DEFAULT_WAIT_SECONDS,
    poll: float = POLL_SECONDS,
    on_wait=None,
) -> Iterator[tuple[object, DeviceProfile]]:
    """Open a free Tenstorrent chip, waiting for one if they are all busy.

    Yields (device, profile) with the program cache enabled, and closes and
    unlocks the chip on the way out. `chip_id` is a local index into the
    visible set, or None to take the first free chip. `on_wait(survey)` is
    called on each poll while blocked.
    """
    # Deferred: this module loads before chip scoping, and ttnn must not be
    # imported until TT_VISIBLE_DEVICES is set.
    import ttnn

    candidates = [as_local_index(chip_id)] if chip_id is not None else enumerate_chips()
    if not candidates:
        raise NoDeviceAvailable(
            "no /dev/tenstorrent/<n> devices found, so there is no chip to "
            "open. Run `ttrl doctor`, which reports whether the driver is "
            "loaded and which chips this process can see."
        )

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        for cid in candidates:
            if status(cid).foreign_pids:
                continue
            with hold_lock(cid) as got:
                if not got:
                    continue
                device = ttnn.open_device(
                    device_id=cid,
                    l1_small_size=cfg.l1_small_size,
                    trace_region_size=cfg.trace_region_size,
                )
                body_exc: BaseException | None = None
                try:
                    try:
                        device.enable_program_cache()
                        yield device, probe_profile(device, cfg)
                    except BaseException as exc:
                        body_exc = exc
                        raise
                finally:
                    # Purge before the close, never after. Both nested,
                    # because neither may cost us the other: a purge that
                    # raises must still close the chip rather than strand it
                    # open for the life of the process, and a raise in a
                    # finally discards whatever the body was unwinding, so a
                    # purge failure is noted on the in-flight exception
                    # instead of replacing it.
                    try:
                        purge_device_caches(device)
                    except BaseException:
                        if body_exc is None:
                            raise
                        body_exc.add_note(
                            "device cache purge also failed:\n"
                            + traceback.format_exc()
                        )
                    finally:
                        ttnn.close_device(device)
                return

        if not wait:
            raise NoDeviceAvailable(
                "every candidate chip is busy:\n"
                + describe_holders(survey())
                + f"\nNote: {STALE_FD_NOTE}"
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise NoDeviceAvailable(
                f"no chip free after {timeout:.0f}s:\n"
                + describe_holders(survey())
                + f"\nNote: {STALE_FD_NOTE}"
            )
        if on_wait is not None:
            on_wait(survey())
        time.sleep(poll)
