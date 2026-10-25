"""What `import ttrl` must not drag in, and what it may cost.

Pinned with a `find_spec`-based `MetaPathFinder`: the legacy `find_module`
hook is ignored from Python 3.12, so a blocker written that way passes while
blocking nothing. Every test here runs in a fresh child, because this process
imported ttrl long ago and a second import is a dict lookup.

The advertised surface itself is pinned in `tests/test_public_api.py`, whose
globbed subpackage list this reads rather than repeating.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.test_public_api import SUBPACKAGES


BLOCKER = '''
import sys


class Blocked(Exception):
    pass


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("ttnn", "gymnasium", "ale_py", "cv2", "mujoco"):
            raise Blocked(f"{fullname} was imported")
        return None


sys.meta_path.insert(0, Blocker())
'''


def _blocked(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(body)],
        capture_output=True, text=True,
    )


def test_the_blocker_blocks():
    """Without this the next two tests pass on a machine with no ttnn at all."""
    result = _blocked("import ttnn")
    assert result.returncode != 0
    assert "ttnn was imported" in result.stderr


def test_importing_ttrl_imports_no_ttnn_and_no_env_suite():
    result = _blocked("import ttrl; print(ttrl.__version__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_the_configuration_surface_resolves_with_ttnn_blocked():
    """Everything a user needs to describe a run before any chip exists."""
    result = _blocked(
        """
        import ttrl
        for name in ("ObsSpec", "ActionSpec", "TrainConfig", "DeviceConfig",
                     "EnvConfig", "load_config", "Trainer", "run", "evaluate",
                     "evaluate_checkpoint", "config_from_checkpoint",
                     "agent_from_checkpoint"):
            getattr(ttrl, name)
        print(ttrl.ObsSpec.vector(4).shape)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(4,)"


def test_reaching_a_subpackage_name_imports_no_ttnn_and_no_env_suite():
    """Naming them all must not cost what importing them would.

    `ttrl.envs` and `ttrl.algos` reach an environment suite and ttnn only when
    something inside them is used, so the attribute itself stays free.
    """
    result = _blocked(
        "import ttrl\n"
        + "\n".join(f"assert ttrl.{n}.__name__ == 'ttrl.{n}'" for n in SUBPACKAGES)
        + "\nprint('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


#: Ceiling on `import ttrl`, in milliseconds. The README reports about 0.1 ms,
#: and this is twenty times that rather than a tight bound on a shared box: the
#: figure is a consequence of the import graph, and what makes it move is a
#: heavy import getting added. The cheapest of those, numpy, costs tens of
#: milliseconds on its own, so a ceiling here separates "imports nothing heavy"
#: from "imports something heavy" without ever failing on load.
IMPORT_CEILING_MS = 2.0


def test_import_ttrl_is_as_cheap_as_the_readme_says():
    """The measured figure is printed and the ceiling is asserted.

    Measured in a clean child, because this process imported ttrl long ago and
    a second import is a dict lookup.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import time; t = time.perf_counter(); import ttrl; "
         "print(f'{(time.perf_counter() - t) * 1e3:.4f}')"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    elapsed = float(result.stdout.strip())
    print(f"\nimport ttrl: {elapsed:.3f} ms")
    assert elapsed < IMPORT_CEILING_MS, (
        f"import ttrl took {elapsed:.3f} ms, over the {IMPORT_CEILING_MS} ms "
        f"ceiling. Something heavy entered the import graph; the tests above "
        f"name the libraries that must stay out of it."
    )
