"""Scoping a process to a chosen set of Tenstorrent chips.

ttnn is never imported here, at module scope or inside a function. A launcher
has to be able to call `scope_to_chip` before the first `import ttnn` in the
process: ttnn reads TT_VISIBLE_DEVICES and TT_MESH_GRAPH_DESC_PATH at
initialisation and never looks again.

Why this exists: `ttnn.open_device(device_id=N)` does not open chip N and leave
the rest alone. UMD maps hugepages and takes the driver mutexes
CHIP_IN_USE_<n>_PCIe for every local chip on the board, so the first process to
open any device blocks every other process from opening any device at all.
TT_VISIBLE_DEVICES makes the whole UMD stack see one chip, and then `device_id`
is a local index into the visible set, so the only legal call is `device_id=0`.
Scoping to one chip of a paired board additionally needs
TT_MESH_GRAPH_DESC_PATH: see MESH_GRAPH_DESC_NAME below. Scoping to a whole
board needs it unset, because ttnn then discovers the real mesh itself and a
forced single-chip descriptor would contradict it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

VISIBLE_DEVICES_ENV = "TT_VISIBLE_DEVICES"
MESH_GRAPH_DESC_ENV = "TT_MESH_GRAPH_DESC_PATH"
#: Required to work around the CUSTOM cluster type produced by splitting a
#: paired board. TT_VISIBLE_DEVICES naming one half of a P300 leaves a cluster
#: shape none of ttnn's own built-in descriptors covers, so it reports
#: `Using CUSTOM cluster type for P300 board with 1 chips` and then fatals at
#: tt_cluster.cpp:285, "Custom fabric mesh graph descriptor path must be
#: specified for CUSTOM cluster type". Measured on ttnn 0.80.0. Despite the
#: p100 name this descriptor says "one chip, no neighbours", which is exactly
#: what the half-board process sees, and it makes that open succeed. The
#: trigger is our own scoping and not any chip count: a process given a whole
#: board needs no descriptor at all.
MESH_GRAPH_DESC_NAME = "p100_mesh_graph_descriptor.textproto"


class ChipScopeError(RuntimeError):
    pass


def mesh_graph_descriptor_path(name: str = MESH_GRAPH_DESC_NAME) -> str | None:
    """Locate `name` inside the installed ttnn package, or return None.

    find_spec locates ttnn without importing it, which matters because the
    caller is usually about to set the environment ttnn reads at import time.
    None is a legitimate answer: the descriptor is needed only on paired
    boards, and the caller decides what its absence means.
    """
    spec = importlib.util.find_spec("ttnn")
    if spec is None:
        return None
    roots = [Path(p) for p in (spec.submodule_search_locations or [])]
    if spec.origin:
        roots.append(Path(spec.origin).parent)
    for root in roots:
        matches = sorted(root.glob(f"**/{name}"))
        if matches:
            return str(matches[0])
    return None


def visible_chip_ids(env=None) -> list[int] | None:
    """Physical chip ids named by TT_VISIBLE_DEVICES, or None if unset.

    None means unscoped (every chip visible); it is not the same as an empty
    set. A malformed value raises: dropping unparseable tokens renumbers the
    visible set, so with "1,x,3" local index 1 stops meaning chip 3's
    neighbour and starts meaning chip 3, putting every replica on the wrong
    chip.
    """
    environ = os.environ if env is None else env
    raw = str(environ.get(VISIBLE_DEVICES_ENV, "")).strip()
    if not raw:
        return None
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok.lstrip("-").isdigit():
            raise ChipScopeError(
                f"{VISIBLE_DEVICES_ENV}={raw!r} is malformed: {tok!r} is not a "
                "chip id. Expected a comma-separated list of integers."
            )
        ids.append(int(tok))
    return ids or None


def physical_chip_id(chip_id: int, env=None) -> int:
    """Map a ttnn-local device_id to the /dev/tenstorrent/<n> it means.

    Out of range raises: passing an untranslated index through means the lock
    file and the fuser probe that follow are about some other chip, which is
    the silent cross-chip aliasing scoping exists to stop.
    """
    visible = visible_chip_ids(env)
    if visible is None:
        return chip_id
    if not 0 <= chip_id < len(visible):
        raise ChipScopeError(
            f"chip id {chip_id} is out of range for {VISIBLE_DEVICES_ENV}="
            f"{visible}: valid local ids are 0..{len(visible) - 1}."
        )
    return visible[chip_id]


def local_index(physical: int, env=None) -> int:
    """Inverse of physical_chip_id: the device_id ttnn will accept."""
    visible = visible_chip_ids(env)
    if visible is None:
        return physical
    if physical not in visible:
        raise ChipScopeError(
            f"physical chip {physical} is not in {VISIBLE_DEVICES_ENV}={visible}"
        )
    return visible.index(physical)


def as_local_index(chip_id: int) -> int:
    """Read a caller-supplied chip_id as something ttnn will accept.

    Identity when unscoped. Scoped, an id still in range is taken as the local
    index the caller computed (the documented contract); otherwise it is
    translated as a visible physical id rather than reaching open_device and
    dying with "Physical chip id N not found in control plane chip mapping".
    """
    visible = visible_chip_ids()
    if visible is None or 0 <= chip_id < len(visible):
        return chip_id
    return local_index(chip_id)


def device_already_open() -> bool:
    """Has this process got a /dev/tenstorrent/* node open? Read from /proc."""
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            # The fd closed between listing and reading, so it is not the
            # open device node being hunted here.
            continue
        if target.startswith("/dev/tenstorrent/"):
            return True
    return False


def chips_env(
    chips: "Sequence[int]", *, base=None, require_descriptor: bool = False
) -> dict[str, str]:
    """The environment that scopes a process to exactly `chips`.

    Returns a new dict; nothing is mutated. Hand it to subprocess as `env=`.

    One chip gets MESH_GRAPH_DESC_ENV pointed at MESH_GRAPH_DESC_NAME, for the
    reason given there. Several chips get it REMOVED, including a value the
    caller already had in `base`: ttnn logs "Using auto discovery to generate
    mesh graph" and reports the real shape only when no descriptor is forced,
    and a stale single-chip descriptor would claim one chip with no neighbours
    on a process holding a whole board.

    With one chip MESH_GRAPH_DESC_ENV is omitted rather than blanked when the
    descriptor cannot be located, so an existing value in `base` survives.
    """
    ids = [int(c) for c in chips]
    if not ids:
        raise ChipScopeError(
            "no chips to scope to; pass at least one physical chip id"
        )
    if len(set(ids)) != len(ids):
        raise ChipScopeError(f"duplicate physical chip ids in {ids}")
    for chip in ids:
        if chip < 0:
            raise ChipScopeError(f"physical chip id must be non-negative, got {chip}")
    env = dict(os.environ if base is None else base)
    env[VISIBLE_DEVICES_ENV] = ",".join(str(c) for c in ids)
    if len(ids) > 1:
        env.pop(MESH_GRAPH_DESC_ENV, None)
        return env
    descriptor = mesh_graph_descriptor_path()
    if descriptor is not None:
        env[MESH_GRAPH_DESC_ENV] = descriptor
    elif require_descriptor and not env.get(MESH_GRAPH_DESC_ENV):
        raise ChipScopeError(
            f"could not locate {MESH_GRAPH_DESC_NAME} in the installed ttnn "
            f"package; set {MESH_GRAPH_DESC_ENV} explicitly"
        )
    return env


def chip_env(chip: int, *, base=None, require_descriptor: bool = False) -> dict[str, str]:
    """`chips_env` for the one-chip case."""
    return chips_env([chip], base=base, require_descriptor=require_descriptor)


def scope_to_chips(
    chips: "Sequence[int]", *, require_descriptor: bool = False
) -> dict[str, str]:
    """Install chips_env into os.environ for this process, before import ttnn.

    Refuses once a device node is open, because the change cannot take effect
    and would only make this process disagree with reality about which chips it
    holds: wrong lock file, wrong fuser probe, wrong chip in the logs.

    MESH_GRAPH_DESC_ENV is removed from os.environ, not merely left out of the
    returned dict, when scoping to several chips: os.environ.update cannot
    unset a variable, and a leftover single-chip descriptor defeats the mesh
    discovery a multi-chip run depends on.
    """
    if "ttnn" in sys.modules and device_already_open():
        raise ChipScopeError(
            f"refusing to set {VISIBLE_DEVICES_ENV}: ttnn is imported and this "
            "process already has a /dev/tenstorrent device open, so the change "
            "cannot take effect. Scope before importing ttnn, or launch a "
            "subprocess with env=chips_env(...)."
        )
    env = chips_env(chips, base={}, require_descriptor=require_descriptor)
    if MESH_GRAPH_DESC_ENV not in env:
        os.environ.pop(MESH_GRAPH_DESC_ENV, None)
    os.environ.update(env)
    return env


def scope_to_chip(chip: int, *, require_descriptor: bool = False) -> dict[str, str]:
    """`scope_to_chips` for the one-chip case."""
    return scope_to_chips([chip], require_descriptor=require_descriptor)
