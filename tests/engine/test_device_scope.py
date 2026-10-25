"""Single-chip scoping."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ttrl.engine import device as device_mod
from ttrl.engine.device import (
    liveness as device_liveness_mod,
    scope as chipscope,
    scope as scope_mod,
)
from ttrl.engine.device.scope import ChipScopeError, chips_env


def test_chip_env_does_not_mutate_the_base_or_environ(monkeypatch):
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    base = {"PATH": "/bin"}
    env = chipscope.chip_env(1, base=base)
    assert env[chipscope.VISIBLE_DEVICES_ENV] == "1"
    assert base == {"PATH": "/bin"}
    assert env["PATH"] == "/bin"
    assert chipscope.VISIBLE_DEVICES_ENV not in os.environ


def test_chip_env_rejects_a_negative_chip():
    with pytest.raises(ChipScopeError):
        chipscope.chip_env(-1, base={})


def test_chip_env_points_at_the_single_chip_mesh_descriptor():
    found = chipscope.mesh_graph_descriptor_path()
    if found is None:
        pytest.skip("ttnn package has no p100 mesh graph descriptor")
    assert Path(found).name == chipscope.MESH_GRAPH_DESC_NAME
    assert Path(found).exists()
    env = chipscope.chip_env(0, base={})
    assert env[chipscope.MESH_GRAPH_DESC_ENV] == found


def test_chip_env_requires_descriptor_only_when_asked(monkeypatch):
    monkeypatch.setattr(chipscope, "mesh_graph_descriptor_path", lambda *a, **k: None)
    env = chipscope.chip_env(0, base={})
    assert chipscope.MESH_GRAPH_DESC_ENV not in env
    with pytest.raises(ChipScopeError):
        chipscope.chip_env(0, base={}, require_descriptor=True)
    # An explicit value in the base env satisfies the requirement.
    kept = chipscope.chip_env(
        0, base={chipscope.MESH_GRAPH_DESC_ENV: "/tmp/x.textproto"}, require_descriptor=True
    )
    assert kept[chipscope.MESH_GRAPH_DESC_ENV] == "/tmp/x.textproto"


def test_scope_to_chip_refuses_when_a_device_is_already_open(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", sys.modules.get("ttnn") or object())
    monkeypatch.setattr(chipscope, "device_already_open", lambda: True)
    with pytest.raises(ChipScopeError):
        chipscope.scope_to_chip(0)


def test_mapping_is_identity_when_unscoped(monkeypatch):
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    assert chipscope.visible_chip_ids() is None
    assert chipscope.physical_chip_id(3) == 3
    assert chipscope.local_index(3) == 3


def test_multiple_visible_chips_map_by_position(monkeypatch):
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "3,1")
    assert chipscope.physical_chip_id(0) == 3
    assert chipscope.physical_chip_id(1) == 1
    assert chipscope.local_index(1) == 1
    assert chipscope.local_index(3) == 0


def test_physical_chip_id_rejects_an_out_of_range_local_index(monkeypatch):
    """An index past the visible set is not translatable, so it must not be passed through."""
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "2")
    assert chipscope.physical_chip_id(0) == 2
    for bad in (1, 2, 7, -1):
        with pytest.raises(ChipScopeError, match="out of range"):
            chipscope.physical_chip_id(bad)


def test_local_index_rejects_an_invisible_chip(monkeypatch):
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "2")
    with pytest.raises(ChipScopeError):
        chipscope.local_index(0)


def test_blank_visible_devices_reads_as_unscoped(monkeypatch):
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "   ")
    assert chipscope.visible_chip_ids() is None
    assert chipscope.physical_chip_id(1) == 1


def test_enumerate_chips_returns_local_indices_when_scoped(monkeypatch):
    monkeypatch.setattr(device_liveness_mod, "physical_chips", lambda: [0, 1, 2, 3])
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    assert device_mod.enumerate_chips() == [0, 1, 2, 3]

    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "2")
    assert device_mod.enumerate_chips() == [0]
    assert device_mod.status(0).path == "/dev/tenstorrent/2"

    # A visible id with no device node is dropped rather than offered.
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "2,9")
    assert device_mod.enumerate_chips() == [0]


def test_acquire_translates_a_physical_chip_id_argument(monkeypatch):
    monkeypatch.setenv(chipscope.VISIBLE_DEVICES_ENV, "2")
    assert scope_mod.as_local_index(0) == 0
    assert scope_mod.as_local_index(2) == 0
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    assert scope_mod.as_local_index(2) == 2


def _local_chip_count() -> int:
    """Chip nodes on this box, counted WITHOUT importing ttnn."""
    visible = os.environ.get("TT_VISIBLE_DEVICES")
    if visible:
        # The env var, not the device nodes, decides what this process (and
        # anything it spawns) can actually open -- the nodes stay on disk
        # either way.
        return len([tok for tok in visible.split(",") if tok.strip()])
    try:
        return len([n for n in os.listdir("/dev/tenstorrent") if n.isdigit()])
    except OSError:
        return 0

_CONCURRENT_PROBE = textwrap.dedent(
    """
    import json, os, sys, time
    from ttrl.engine.device.scope import scope_to_chip
    scope_to_chip(int(sys.argv[1]))
    import ttnn
    from ttrl.engine import device as device_mod
    with device_mod.acquire(chip_id=0, wait=False) as (dev, profile):
        time.sleep(float(sys.argv[2]))
        # Sentinel-prefixed: ttnn logs device teardown to stdout AFTER this
        # block exits, so the JSON is not reliably the last line.
        print("TTRL_PROBE " + json.dumps({"visible": os.environ["TT_VISIBLE_DEVICES"],
                                          "arch": profile.arch}))
    """
)


@pytest.mark.tt
@pytest.mark.skipif(
    _local_chip_count() < 4,
    reason=(
        "needs 2 chips FREE OF THIS PROCESS: the parent pytest process has "
        "imported ttnn, so UMD holds descriptors on every chip visible to it "
        "and the two spawned children have none left to scope to. Run with at "
        "least 4 chips visible (TT_VISIBLE_DEVICES unset on a 4-chip box)."
    ),
)

def test_two_scoped_processes_acquire_two_chips_without_contention(tmp_path):
    """Two chips, two processes, zero waiting -- the whole point of scoping."""

    free = [st.chip_id for st in device_mod.survey() if st.free]
    if len(free) < 2:
        pytest.skip(f"need two free chips, survey says {free}")
    a, b = free[:2]

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CONCURRENT_PROBE, str(chip), "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"},
        )
        for chip in (a, b)
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    for (out, err), proc in zip(outs, procs):
        assert proc.returncode == 0, f"stdout={out}\nstderr={err}"
    def _probe_json(out: str) -> dict:
        for line in out.splitlines():
            if line.startswith("TTRL_PROBE "):
                return json.loads(line[len("TTRL_PROBE "):])
        raise AssertionError(f"no TTRL_PROBE line in probe stdout:\n{out}")

    visible = sorted(_probe_json(out)["visible"] for out, _ in outs)
    assert visible == sorted([str(a), str(b)])


def test_chips_env_drops_the_descriptor_for_a_multi_chip_scope():
    """A whole-board process must reach ttnn with no descriptor forced."""
    env = chips_env([2, 3], base={chipscope.MESH_GRAPH_DESC_ENV: "/stale.textproto"})
    assert env[chipscope.VISIBLE_DEVICES_ENV] == "2,3"
    assert chipscope.MESH_GRAPH_DESC_ENV not in env


def test_chips_env_rejects_an_empty_or_duplicated_set():
    with pytest.raises(ChipScopeError, match="at least one"):
        chips_env([], base={})
    with pytest.raises(ChipScopeError, match="[Dd]uplicate"):
        chips_env([3, 3], base={})


def test_scope_to_chips_unsets_a_stale_descriptor_in_this_process(monkeypatch):
    """os.environ.update cannot remove a variable, so the multi-chip path has to pop it explicitly."""
    monkeypatch.setattr(chipscope, "device_already_open", lambda: False)
    monkeypatch.setenv(chipscope.MESH_GRAPH_DESC_ENV, "/stale.textproto")
    # Recorded before `scope_to_chips` writes it, so the caller's scope is
    # restored at teardown. Leaking it rescopes every later test in this
    # process, and a child they spawn then opens a chip nobody asked for.
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    chipscope.scope_to_chips([2, 3])
    assert os.environ[chipscope.VISIBLE_DEVICES_ENV] == "2,3"
    assert chipscope.MESH_GRAPH_DESC_ENV not in os.environ


def test_scope_to_chips_keeps_the_descriptor_for_one_chip(monkeypatch):
    monkeypatch.setattr(chipscope, "device_already_open", lambda: False)
    monkeypatch.delenv(chipscope.MESH_GRAPH_DESC_ENV, raising=False)
    monkeypatch.delenv(chipscope.VISIBLE_DEVICES_ENV, raising=False)
    chipscope.scope_to_chips([3])
    assert os.environ[chipscope.VISIBLE_DEVICES_ENV] == "3"
    assert os.environ[chipscope.MESH_GRAPH_DESC_ENV].endswith(
        chipscope.MESH_GRAPH_DESC_NAME
    )
