"""Which algorithms have a step mode, and therefore need a parity gate.

`tests/hardware/test_determinism_fingerprint.py` runs each covered algorithm
once per mode in `ttrl.engine.step.MODES` and requires one fingerprint per
algorithm. It covers DQN and SAC, which are the algorithms with a mode to get
wrong: PPO calls the engine directly and has no `step_mode` field.

That sentence is the kind that goes stale. So the set is enumerated here
rather than believed: the algorithm packages are globbed off disk and their
config modules read for the field. The day a second algorithm gains one, this
test fails and names it, which is the prompt to widen the parity gate instead
of discovering the gap from a training anomaly.

Host, and deliberately import-free: a config module must not import ttnn
(`tests/test_algos_no_ttnn.py`), and reading the source with `ast` cannot
break that promise by accident.
"""

from __future__ import annotations

import ast
from pathlib import Path

ALGOS_DIR = Path(__file__).resolve().parents[1] / "ttrl" / "algos"

#: Algorithms whose eager-vs-traced parity the hardware fingerprint gate runs,
#: each with its own fingerprint and its own `agrees_bit_for_bit` test in
#: `tests/hardware/test_determinism_fingerprint.py`. The test below is what
#: keeps this set honest.
COVERED_BY_THE_PARITY_GATE = {"dqn", "sac"}

FIELD = "step_mode"


def _algo_packages() -> dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(ALGOS_DIR.glob("*/config.py"))
    }


def _declares_field(path: Path, field: str) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.target.id == field):
                    return True
    return False


def test_there_are_algorithm_packages_to_enumerate():
    """A glob that matches nothing would make the test below vacuous."""
    packages = _algo_packages()
    assert len(packages) >= 3, packages


def test_every_algorithm_with_a_step_mode_has_a_parity_gate():
    with_mode = {name for name, path in _algo_packages().items()
                 if _declares_field(path, FIELD)}
    assert with_mode == COVERED_BY_THE_PARITY_GATE, (
        f"algorithms declaring {FIELD}: {sorted(with_mode)}; the eager-vs-traced "
        f"parity gate covers {sorted(COVERED_BY_THE_PARITY_GATE)}. Add the new "
        f"one to tests/hardware/test_determinism_fingerprint.py and to the set "
        f"above, or take the field away again. An untested step mode is a "
        f"second numeric path nothing compares against the first."
    )
