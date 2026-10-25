"""`ttrl.buffers` must import and work with no ttnn and no chip.

The sampling semantics in this package are the easiest thing in the repo to get
silently wrong, so they have to be testable on a machine with no Tenstorrent
hardware. That is only true as long as nothing in the package reaches for ttnn,
and an accidental `import ttnn` would not fail here: the real wheel is
installed, so it would simply succeed and quietly move these tests into the
hardware half of the suite.

Both checks therefore poison ttnn rather than merely look for it, and each one
proves its own poison is live before trusting the result. `find_module` is
deliberately not used: it was removed in Python 3.12, so a finder relying on it
is never consulted and the check would pass by doing nothing.
"""

from __future__ import annotations

import importlib
import importlib.abc
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ttrl.buffers import ReplayBuffer, RolloutBuffer
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec


class _RefuseTtnn(importlib.abc.MetaPathFinder):
    """Raises if anything asks to import ttnn. `find_spec`, not `find_module`."""

    def __init__(self):
        self.asked = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ttnn" or fullname.startswith("ttnn."):
            self.asked.append(fullname)
            raise AssertionError(f"ttrl.buffers imported {fullname}")
        return None


def _buffer_modules() -> list[str]:
    """Every module of `ttrl.buffers`, walked rather than listed.

    The list that used to stand here named five of the eleven modules in the
    package, so `persist`, `sampling`, `sequences`, `seqbatch`, `boundaries`
    and `replaybatch` were covered by the docstring above and by nothing else.
    The claim is about the package, so the set is the package.

    The names are collected off disk without importing anything, because the
    import is what the tests below are measuring.
    """
    package = pathlib.Path(__file__).resolve().parents[2] / "ttrl" / "buffers"
    names = ["ttrl.buffers"] + sorted(
        f"ttrl.buffers.{path.stem}" for path in package.glob("*.py")
        if path.stem != "__init__"
    )
    assert len(names) > 6, names
    return names


BUFFER_MODULES = _buffer_modules()


def test_buffers_import_with_a_find_spec_finder_that_refuses_ttnn():
    finder = _RefuseTtnn()
    sys.meta_path.insert(0, finder)
    try:
        for name in BUFFER_MODULES:
            sys.modules.pop(name, None)
        for name in BUFFER_MODULES:
            importlib.import_module(name)
        assert finder.asked == []
        # The finder is live: asking for a ttnn name through it must raise.
        # A submodule nothing has imported, because `import_module("ttnn")`
        # answers from sys.modules without consulting meta_path once any
        # earlier test in the session has imported ttnn, which made this
        # self-check pass or fail on test order.
        with pytest.raises(AssertionError):
            importlib.import_module("ttnn._not_a_real_submodule")
    finally:
        sys.meta_path.remove(finder)
        for name in BUFFER_MODULES:
            sys.modules.pop(name, None)
        for name in BUFFER_MODULES:
            importlib.import_module(name)


_SUBPROCESS_CHECK = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    sys.path.insert(1, sys.argv[2])
    import importlib
    for name in sys.argv[3].split(","):
        importlib.import_module(name)
    assert "ttnn" not in sys.modules, "ttnn was imported"
    try:
        import ttnn
    except AssertionError as exc:
        print("STUB-LIVE", exc)
    else:
        print("STUB-DEAD")
        raise SystemExit(2)
    print("OK")
    """
)


def test_buffers_import_in_an_isolated_subprocess_with_a_stub_ahead_of_the_wheel(tmp_path):
    """A second, independent method: a poisoned `ttnn.py` earlier on sys.path.

    Run with `-I`, so neither the user's site-packages nor any inherited
    PYTHONPATH can reorder the stub behind the real wheel.
    """
    (tmp_path / "ttnn.py").write_text(
        'raise AssertionError("ttrl.buffers must not import ttnn")\n'
    )
    repo_root = str(pathlib.Path(__file__).resolve().parents[2])
    script = tmp_path / "check.py"
    script.write_text(_SUBPROCESS_CHECK)
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(tmp_path), repo_root,
         ",".join(BUFFER_MODULES)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STUB-LIVE" in result.stdout, result.stdout
    assert "OK" in result.stdout, result.stdout


def test_the_whole_buffer_api_works_without_ttnn_imported():
    """Not just importable: usable. Driven after the import checks above."""

    obs = ObsGroup.of(ObsSpec.vector(3))
    act = ActionSpec.discrete(2)
    replay = ReplayBuffer(obs, act, capacity=32, n_envs=2, n_step=2, gamma=0.9, seed=0)
    for t in range(10):
        replay.add_batch(np.zeros((2, 3), np.float32), np.zeros(2, np.int64),
                         np.ones(2, np.float32), np.zeros(2, bool))
    assert len(replay.sample(8)) == 8

    rollout = RolloutBuffer(obs, act, n_envs=2, n_steps=2, seed=0)
    for _ in range(2):
        rollout.add(np.zeros((2, 3), np.float32), np.zeros(2, np.int64),
                    np.zeros(2, np.float32), np.zeros(2, np.float32),
                    np.ones(2, np.float32), np.zeros(2, np.float32))
    rollout.compute_gae(0.99, 0.95, np.zeros(2, np.float32))
    assert len(list(rollout.minibatches(2, epochs=1))) == 2
    # Deliberately no `"ttnn" not in sys.modules` here. Several other modules
    # in this suite import ttnn at module scope, so in a full run that would
    # assert something about collection order rather than about this package.
    # Whether the buffers reach for ttnn is settled by the two tests above,
    # each of which poisons the import inside its own scope.
