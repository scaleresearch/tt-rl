"""Reaching a config must not import ttnn.

`ttrl.load_config` runs before any chip is scoped, and a run's config file
constructs `DQNConfig(...)` at import time. A process that touches ttnn before
TT_VISIBLE_DEVICES has been applied does not open one chip: it maps hugepages
and takes the driver mutex for every chip on the board, so a run can end up
looking at chips it does not own.

Each case runs in a fresh interpreter with a `find_spec` import hook that
refuses "ttnn". In-process assertions would be worthless here: the rest of this
suite imports ttnn, so `sys.modules` already holds it by the time any test
body runs. The hook implements `find_spec` and not the legacy `find_module`,
which Python 3.12 removed and silently ignores.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ttrl.algos

REPO = Path(__file__).resolve().parent.parent

#: Installed before ttrl is imported, so any ttnn import raises rather than
#: loading. Refuses submodules too, since `import ttnn.operations` would
#: initialise the package.
_HOOK = """
import sys


class RefuseTtnn:
    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            raise AssertionError(f"{name} was imported")
        return None


sys.meta_path.insert(0, RefuseTtnn())
"""

_CHECK = """
assert "ttnn" not in sys.modules, "ttnn reached sys.modules"
print("OK")
"""

#: `config class name -> (its own package, its agent class)`, derived from
#: `ttrl.algos.__all__` and the algorithm packages on disk. The list that used
#: to stand here named dqn, ppo and sac: an algorithm added with a lazy config
#: would have gone unchecked, and one added correctly would still have needed
#: this file edited before it passed. Importing `ttrl.algos` here is safe and
#: is the thing under test elsewhere in this file: the package is lazy, and
#: every case below runs in a subprocess regardless.
def _algorithms() -> dict[str, tuple[str, str]]:
    packages = {path.parent.name
                for path in (REPO / "ttrl" / "algos").glob("*/config.py")}
    found = {}
    for name in ttrl.algos.__all__:
        # `AlgoConfig` is the contract every algorithm's config satisfies, so
        # it belongs to no package and is checked by its own case below.
        if not name.endswith("Config") or name == "AlgoConfig":
            continue
        package = name[: -len("Config")].lower()
        assert package in packages, (
            f"{name} is exported by ttrl.algos but there is no "
            f"ttrl/algos/{package}/config.py; the packages on disk are "
            f"{sorted(packages)}"
        )
        found[name] = (package, name[: -len("Config")])
    assert sorted(p for p, _cls in found.values()) == sorted(packages), (
        f"algorithm packages on disk are {sorted(packages)} and the ones "
        f"ttrl.algos exports a config for are "
        f"{sorted(p for p, _cls in found.values())}. A config that cannot be "
        f"reached through the package cannot be reached before a chip is "
        f"scoped either."
    )
    return found


ALGORITHMS = _algorithms()

CASES = {
    **{package: f"from ttrl.algos.{package} import {config}; {config}()"
       for config, (package, _cls) in ALGORITHMS.items()},
    **{f"{package} via package":
       f"from ttrl.algos import {config}; {config}()"
       for config, (package, _cls) in ALGORITHMS.items()},
    # The train layer calls this from TrainConfig.__post_init__, so the
    # refusal has to fire before a chip is opened or an environment built.
    "scaling": (
        "from ttrl.algos.scaling import check_network_batch, divide_for_replicas\n"
        "from ttrl.algos.dqn import DQNConfig\n"
        "cfg = DQNConfig(batch_size=8)\n"
        "check_network_batch(cfg, 4, 'dqn')\n"
        "assert cfg.scale_for_replicas(1) is cfg"
    ),
    # ASKING WHICH ALGORITHMS EXIST MUST IMPORT NONE OF THEM. `ttrl.train.run`
    # asks this before `scope_for_run` has set TT_VISIBLE_DEVICES, and a
    # third-party package guards its own registration with exactly this
    # expression, because it is what this library's own code looks like. A
    # registry that imported its builders to answer would map hugepages and
    # take the driver mutex for every chip on the board, with no error
    # anywhere to say so.
    "algos registry membership": (
        "from ttrl.algos import ALGOS\n"
        "assert 'dqn' in ALGOS\n"
        "assert 'mine' not in ALGOS\n"
        "assert ALGOS.names() == ['dqn', 'ppo', 'sac'], ALGOS.names()\n"
        "assert len(ALGOS) == 3\n"
        "ALGOS.register('mine', lambda *a, **k: None)\n"
        "assert 'mine' in ALGOS\n"
        "assert ALGOS.entry('dqn') == 'ttrl.algos.dqn.agent:DQN'"
    ),
    # Same question through the public surface, which is the spelling an
    # outside author actually writes.
    "algos registry membership via ttrl": (
        "import ttrl\n"
        "assert 'mine' not in ttrl.ALGOS\n"
        "ttrl.ALGOS.register('mine', lambda *a, **k: None)\n"
        "assert sorted(ttrl.ALGOS.names()) == ['dqn', 'mine', 'ppo', 'sac']"
    ),
    # The algo-config contract and its checker: a third-party config is
    # validated by `load_config` before any chip is scoped, so neither the
    # protocol nor the checker may drag ttnn in.
    "algo config contract": (
        "import dataclasses\n"
        "from ttrl.algos.config import AlgoConfig, check_algo_config\n"
        "from ttrl.train.checkpoint.types import register_checkpoint_type\n"
        "@register_checkpoint_type\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class Mine:\n"
        "    NETWORK_BATCH_FIELD = 'batch_size'\n"
        "    batch_size: int = 8\n"
        "    @property\n"
        "    def network_batch(self): return self.batch_size\n"
        "    def scale_for_replicas(self, n): return self\n"
        "check_algo_config(Mine(), 'mine')\n"
        "assert isinstance(Mine(), AlgoConfig)"
    ),
    # The replay-ratio clock is pure arithmetic and the one piece of the
    # off-policy cadence a host test can check exactly, so it must not drag
    # ttnn in behind it.
    "the update clock": (
        "from ttrl.algos.clock import UpdateClock\n"
        "c = UpdateClock(2, 0); c.advance(8)\n"
        "assert c.due and c.owed == 4"
    ),
    "the agent contract": (
        "from ttrl.algos import ALGOS, Agent, Transition, check_agent\n"
        "assert ALGOS.names  # the registry resolves builders lazily"
    ),
    "the whole package": "import ttrl, ttrl.algos",
}


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _HOOK + body + "\n" + _CHECK],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("name", list(CASES))
def test_reaching_a_config_needs_no_ttnn(name):
    done = _run(CASES[name])
    assert done.returncode == 0, (
        f"{name!r} imported ttnn or failed:\n{done.stdout}\n{done.stderr}"
    )
    assert "OK" in done.stdout


def test_the_hook_really_refuses_ttnn():
    """The control. Without it, every case above passes vacuously if the hook
    is wrong, which is the usual way this test goes quietly dead."""
    done = _run("import ttnn")
    assert done.returncode != 0
    assert "ttnn was imported" in done.stderr


def test_an_agent_still_needs_ttnn_and_says_so_through_the_hook():
    """The other control: the agents genuinely do pull ttnn, so the lazy
    surface is load-bearing rather than decorative. One per algorithm, from
    the same derived table: an agent that stopped needing ttnn would mean the
    lazy surface above is testing nothing."""
    for package, cls in sorted(ALGORITHMS.values()):
        body = f"from ttrl.algos.{package} import {cls}"
        done = _run(body)
        assert done.returncode != 0, f"{body!r} unexpectedly needed no ttnn"
        assert "was imported" in done.stderr
