"""Scoping the process to the chip a run will use, before ttnn loads.

This is the first thing a run does, and it has to be, which is the whole
difficulty. `ttnn.open_device(device_id=N)` does not open chip N and leave the
rest alone: UMD maps hugepages and takes the driver mutex for every chip on
the board, so the first process to open any device blocks every other process
from opening any. `TT_VISIBLE_DEVICES` is what makes the stack see one chip,
and ttnn reads it at initialisation and never looks again. So it cannot be set
inside `acquire`, which runs after ttnn is loaded; it has to be set here, at
the top of the entry point.

Without this, `ttrl train` takes a four-chip box for one run. That is the
failure `ttrl doctor` warns about in its own output, and it is what stopped
tt-rl being handed to a team sharing a machine.

ONE NOTION OF CHIP IDENTITY. `DeviceConfig.chip` and `DeviceConfig.chips` are
PHYSICAL ids: the number in `/dev/tenstorrent/<n>`, which is what `ttrl
doctor` prints and what a user reads off it. The local index into the visible
set, which is what ttnn's own `device_id` means, is not in the config at all:
after scoping, the visible set holds one chip and the only legal local index
is 0, so a second field would carry no information and two spellings of "which
chip" is what has bitten every reader of this code so far. `acquire` is
therefore always called without an id, meaning "the first free chip I can
see".
"""

from __future__ import annotations

from ttrl.engine.device import physical_chips, scope_to_chip, status, visible_chip_ids

__all__ = ["resolved_chip", "scope_for_run"]


def scope_for_run(device_config) -> int | None:
    """Scope this process to the chip `device_config` asks for.

    Returns the physical chip now in effect, or None when the scope was left
    alone. Safe to call more than once: the entry points are `ttrl.run` and
    `Trainer.__init__`, and one can be reached through the other.

    Multi-chip runs are left alone. `scope_to_chip` scopes to exactly one
    chip, and a run opening several needs the visible set to name all of them;
    `open_independent_devices_for_chips` checks that itself, before ttnn is
    imported, and says what to set.
    """

    if device_config.chips != 1:
        return None

    visible = visible_chip_ids()
    wanted = device_config.chip

    if visible is None:
        chip = _first_free_chip() if wanted is None else wanted
        if chip is None:
            return None
        scope_to_chip(chip, require_descriptor=True)
        return chip

    # Already scoped by the caller, which is a deployment fact and wins over a
    # silent re-interpretation of it.
    if wanted is None:
        return visible[0] if visible else None
    if wanted not in visible:
        raise ValueError(
            f"device.chip is {wanted} but this process is scoped to chip(s) "
            f"{visible} by TT_VISIBLE_DEVICES, which it cannot see past. Set "
            f"device.chip to one of {visible}, or unset TT_VISIBLE_DEVICES and "
            f"let the run scope itself."
        )
    if visible != [wanted]:
        # Narrowing a wider scope to the one chip this run asked for.
        scope_to_chip(wanted, require_descriptor=True)
    return wanted


def _first_free_chip() -> int | None:
    """The lowest chip nobody else is holding, or the lowest chip at all.

    Falling back to a held chip rather than refusing: scoping to a contended
    chip still takes one driver mutex instead of every one on the board, which
    is strictly better for everyone else, and `acquire` does the real waiting
    with a bounded timeout and a message naming the holder.
    """

    chips = physical_chips()
    if not chips:
        return None
    return next((c for c in chips if status(c).free), chips[0])


def resolved_chip(device_config) -> int | None:
    """The physical chip a single-chip run is on, read without scoping.

    Read-only, unlike `scope_for_run`: a caller that built its own `Trainer`
    has a device open already, and setting `TT_VISIBLE_DEVICES` then cannot
    take effect. None for a multi-chip run, whose record names its chips
    through `device.chips`, and None for a process nobody has scoped, where
    the chip the stack will pick is not yet decided.
    """

    if device_config.chips != 1:
        return None
    visible = visible_chip_ids()
    if not visible:
        return None
    if len(visible) == 1:
        return visible[0]
    wanted = device_config.chip
    return wanted if wanted in visible else None
