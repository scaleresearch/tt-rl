"""Chip resolution and the visible-scope guard.

A topology is never inferred from a device count or a filename, and it is no
longer tabulated either: ttnn discovers the physical mesh itself, so the shape
is read off the hardware by `ttrl.engine.mesh.open`. What is left here is
which physical chips a run may touch.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ttrl.engine.device import (
    VISIBLE_DEVICES_ENV,
    local_index,
    physical_chips,
    visible_chip_ids,
)

__all__ = [
    "ParamSyncError",
    "local_ids_for",
    "require_exact_visible_scope",
    "resolve_chips",
]


class ParamSyncError(RuntimeError):
    """Chip resolution failed, or a collective failed mid-run.

    Fatal, not retryable: by the time this propagates out of
    `ParamSync.average` the replicas may be partially averaged.
    """


def resolve_chips(chips, discovered: Sequence[int]) -> list[int]:
    """Resolve "all", an int count, or an explicit id list against discovered
    physical chip ids. An explicit list is validated to be a subset of what
    was discovered, never trusted as given."""
    discovered = sorted(set(discovered))
    if chips == "all":
        if not discovered:
            raise ParamSyncError(
                "chips: all discovered no chips. Check that /dev/tenstorrent nodes "
                f"exist and that {VISIBLE_DEVICES_ENV} does not exclude all of them."
            )
        return discovered
    if isinstance(chips, bool):
        raise ParamSyncError(f"chips must not be a bool, got {chips!r}")
    if isinstance(chips, int):
        if chips < 1:
            raise ParamSyncError(f"chips must be >= 1, got {chips}")
        if chips > len(discovered):
            raise ParamSyncError(
                f"requested {chips} chips but only {len(discovered)} were discovered "
                f"({discovered})."
            )
        return discovered[:chips]
    if isinstance(chips, (list, tuple)):
        wanted = [int(c) for c in chips]
        if len(set(wanted)) != len(wanted):
            raise ParamSyncError(f"duplicate chip ids in {wanted}")
        missing = [c for c in wanted if c not in discovered]
        if missing:
            raise ParamSyncError(
                f"requested chips {missing} are not among the discovered chips "
                f"{discovered}."
            )
        return wanted
    raise ParamSyncError(f"unrecognised chips config: {chips!r}")


def check_visible_scope(chips: Sequence[int], usable: Iterable[int]) -> None:
    """Refuse to open chips from a process whose visible set is wider than the
    chips it wants.

    Fatal rather than advisory: UMD topology discovery takes a file-descriptor
    lock on every visible device node at `import ttnn`, not just the ones later
    passed to CreateDevices. A job on chips [0, 1] launched unscoped therefore
    holds [2, 3] hostage too, and a second job on [2, 3] starves.
    TT_VISIBLE_DEVICES is read at ttnn import time and never re-read, so this
    cannot be repaired from inside the process.

    A job that uses all visible chips is the equality case and passes; this
    fires only on a strict superset.
    """
    requested = {int(c) for c in chips}
    usable = {int(c) for c in usable}
    missing = sorted(requested - usable)
    if missing:
        raise ParamSyncError(
            f"requested chips {missing} are not usable by this process (usable: "
            f"{sorted(usable)}). Check {VISIBLE_DEVICES_ENV} and which "
            "/dev/tenstorrent nodes exist."
        )
    extra = sorted(usable - requested)
    if extra:
        wanted = ",".join(str(c) for c in sorted(requested))
        raise ParamSyncError(
            f"this process can see chips {sorted(usable)} but was asked to run on "
            f"{sorted(requested)}. Importing ttnn takes a driver lock on EVERY visible "
            f"/dev/tenstorrent node, so this run would starve any job on chips {extra} "
            f"even though it never opens them. Relaunch with {VISIBLE_DEVICES_ENV}="
            f"{wanted} set before python starts: ttnn reads it at import time and never "
            "looks again, so it cannot be set from inside this process."
        )


def require_exact_visible_scope(chips: Sequence[int]) -> None:
    visible = visible_chip_ids()
    check_visible_scope(chips, physical_chips() if visible is None else visible)


def local_ids_for(chips: Sequence[int]) -> list[int]:
    """Translate physical chip ids to the device_ids ttnn accepts.

    Via `local_index` only. A "prefer the local reading when in range"
    heuristic is wrong here, where the contract is unambiguously physical ids:
    with TT_VISIBLE_DEVICES=1,2 and chips [1, 2] it maps both to index 1 and
    two replicas silently land on one chip.
    """
    ids = [local_index(int(c)) for c in chips]
    if len(set(ids)) != len(ids):
        raise ParamSyncError(
            f"chips {list(chips)} map to duplicate ttnn device ids {ids}. Refusing to "
            "place two replicas on one chip."
        )
    return ids
