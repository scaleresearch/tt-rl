"""`import ttrl` and `import ttrl.train` must cost nothing."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

from tests.train._extras import (EVERYTHING, EXTRAS, IMPORT_NAMES, SUITES,
                                 SUITE_EXTRAS)
from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.checkpoint import L1, save_checkpoint
from ttrl.train.config import TrainConfig

BLOCKER = '''
import sys


class Blocked(Exception):
    pass


class Blocker:
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise Blocked(f"{fullname} was imported")
        return None


BLOCKED = %(blocked)s
for name in BLOCKED:
    sys.modules.pop(name, None)
sys.meta_path.insert(0, Blocker(BLOCKED))
'''

def test_every_optional_distribution_has_an_import_name():
    """So an extra added to pyproject.toml is blocked on the day it is added
    rather than on the day someone remembers this list."""
    declared = {dist for dists in EXTRAS.values() for dist in dists}
    assert declared == set(IMPORT_NAMES), (
        f"pyproject optional-dependencies hold {sorted(declared)} and this "
        f"file maps {sorted(IMPORT_NAMES)}. An optional package with no "
        f"import name here is one `import ttrl` may pull in untested."
    )
    assert set(SUITE_EXTRAS) <= set(EXTRAS), sorted(EXTRAS)


def _run(body: str, blocked=EVERYTHING) -> subprocess.CompletedProcess:
    prelude = BLOCKER % {"blocked": repr(tuple(blocked))}
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_whole_train_surface_resolves_without_ttnn():
    result = _run(
        """
        import ttrl.train as t
        for name in t.__all__:
            getattr(t, name)
        cfg = t.TrainConfig(total_frames=10, log_dir="runs/x")
        print(cfg.env.n_envs)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_scoping_needs_no_ttnn():
    """The ordering constraint, as a test."""
    result = _run(
        """
        import os
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        import ttrl.train as t
        print(t.scope_for_run(t.DeviceConfig(chip=2)), os.environ["TT_VISIBLE_DEVICES"])
        """,
        blocked=tuple(b for b in EVERYTHING if b != "ttnn"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2 2"


def test_reading_a_checkpoint_imports_no_env_suite(tmp_path):
    """The claim the old `_num_actions` broke: a checkpoint is self-describing."""

    path = tmp_path / "ckpt.npz"
    save_checkpoint(
        str(path),
        agent_state={"online": {"fc.W": np.ones((2, 3), np.float32)}, "t": 7},
        obs_spec=ObsSpec.image(4, 84, 84),
        action_spec=ActionSpec.discrete(6),
        level=L1,
        frame=4242,
        config=TrainConfig(total_frames=10_000, log_dir="runs/x"),
        env_policy_state=[],
    )

    result = _run(
        f"""
        from ttrl.train.restore import agent_from_checkpoint, config_from_checkpoint
        from ttrl.train.checkpoint import load_checkpoint
        import ttrl.train.builders
        import ttrl.train.evaluate

        loaded = load_checkpoint({str(path)!r})
        config = config_from_checkpoint({str(path)!r})
        print(loaded["action_spec"].n, loaded["obs_spec"]["obs"].shape[0],
              loaded["frame"], config.total_frames)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6 4 4242 10000"


def test_the_agent_builder_imports_no_env_suite():
    """Rebuilding an agent reaches ttrl.nets and ttrl.engine, never a suite."""
    result = _run(
        """
        import ttrl.algos.dqn.agent
        import ttrl.algos.ppo.agent
        from ttrl.train.builders import build_agent
        print("ok", build_agent.__name__)
        """,
        blocked=SUITES,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok build_agent"


def test_an_unregistered_env_name_is_refused_without_ttnn(tmp_path):
    """The cheap half of the check, in the place that has to stay ttnn-free."""
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from ttrl.train.config import TrainConfig, EnvConfig\n"
        "config = TrainConfig(total_frames=10, log_dir='runs/x',\n"
        "                     env=EnvConfig(name='cartpole'))\n"
    )
    result = _run(
        f"""
        import ttrl
        try:
            ttrl.load_config({str(cfg)!r})
        except ValueError as exc:
            print("REFUSED", "cartpole" in str(exc), "register" in str(exc))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REFUSED True True"


def test_load_config_does_not_validate_the_algo_name(tmp_path):
    """The other half of the split, and the reason it is a split."""
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from ttrl.train.config import TrainConfig, EnvConfig\n"
        "config = TrainConfig(algo='rainbow', total_frames=10,\n"
        "                     log_dir='runs/x', env=EnvConfig(name='atari'))\n"
    )
    result = _run(
        f"""
        import ttrl
        print(ttrl.load_config({str(cfg)!r}).algo)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rainbow"

