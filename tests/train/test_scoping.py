"""`scope_for_run`: a run takes one chip, not the board."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

PRELUDE = """
import os
os.environ.pop("TT_VISIBLE_DEVICES", None)
os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
from ttrl.train.config import DeviceConfig
from ttrl.train.scoping import scope_for_run
"""


def _run(body: str, *, visible: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("TT_VISIBLE_DEVICES", None)
    scoped = "" if visible is None else f'os.environ["TT_VISIBLE_DEVICES"] = {visible!r}\n'
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + scoped + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def _ok(result) -> str:
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_an_unscoped_run_scopes_itself_to_the_chip_it_names():
    body = """
    chip = scope_for_run(DeviceConfig(chip=2))
    print(chip, os.environ["TT_VISIBLE_DEVICES"])
    """
    assert _ok(_run(body)) == "2 2"


def test_a_chip_outside_the_visible_scope_is_refused():
    body = """
    try:
        scope_for_run(DeviceConfig(chip=2))
    except ValueError as exc:
        print("refused:", exc)
    """
    assert "cannot see past" in _ok(_run(body, visible="1"))


def test_a_multi_chip_run_is_left_to_its_own_visible_scope():
    body = """
    print(scope_for_run(DeviceConfig(chip=None, chips=[1, 2])),
          os.environ["TT_VISIBLE_DEVICES"])
    """
    assert _ok(_run(body, visible="1,2")) == "None 1,2"


def test_scoping_with_a_device_open_is_refused_rather_than_silently_ignored():
    """The ordering constraint, as an error."""
    body = """
    import sys, types
    sys.modules["ttnn"] = types.ModuleType("ttnn")
    import ttrl.engine.device.scope as s
    s.device_already_open = lambda: True
    try:
        scope_for_run(DeviceConfig(chip=2))
    except s.ChipScopeError as exc:
        print("refused:", exc)
    """
    assert "cannot take effect" in _ok(_run(body))
