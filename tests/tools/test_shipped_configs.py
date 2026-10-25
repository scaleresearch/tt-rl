"""Every shipped config and every example still builds, on the host.

Nothing in the suite called `load_config` on a config that ships. `configs/*.py`
were loaded only under the hardware marker, and no test imported `examples/`
at all, so when `DeviceConfig.chip_id` was renamed to `chip` the host gate
stayed green while every shipped config, every example and every sweep died
at config load. This is the cheapest test in the tree and it would have
caught that for free.

BOTH LISTS ARE GLOBBED, never named. A hand-maintained list of configs drifts
from what ships, which is the same failure one level up: `GAMES` in
`tests/hardware/test_configs_smoke.py` sampled three of the ten shipped
modules, so "Pong passes" was never true, Pong was simply never run.

Lives under `tests/tools/` because the sweep launcher is the other thing that
breaks when a config field is renamed, and the two are checked together.
Needs no hardware: `load_config` builds and validates dataclasses, imports no
ttnn, and opens nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ttrl.train.config import TrainConfig, load_config

_ROOT = Path(__file__).resolve().parents[2]

#: Config modules that ship with the library, globbed.
_CONFIGS = sorted(p for p in (_ROOT / "configs").glob("*.py") if not p.name.startswith("_"))

#: Example projects, globbed. Each exports `parse_args` and
#: `build_config(args)`, which is the example's config: the examples build a
#: `TrainConfig` literal in code rather than importing one from `configs/`.
_EXAMPLES = sorted((_ROOT / "examples").glob("*/train.py"))


def test_there_are_configs_and_examples_to_check():
    """A glob that matches nothing would make every test below vacuous."""
    assert len(_CONFIGS) >= 10, _CONFIGS
    assert len(_EXAMPLES) >= 2, _EXAMPLES


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_a_shipped_config_loads_and_takes_the_overrides_a_sweep_appends(path):
    config = load_config(
        str(path), overrides=["seed=3", "env.seed=3", "log_dir=runs/x", "device.chip=1"]
    )
    assert isinstance(config, TrainConfig)
    assert config.seed == 3 and config.env.seed == 3
    assert config.log_dir == "runs/x"
    assert config.device.chip == 1


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.parent.name)
def test_an_example_builds_its_config_from_its_own_defaults(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path)])
    spec = importlib.util.spec_from_file_location(f"ttrl_example_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        config = module.build_config(module.parse_args())
    finally:
        del sys.modules[spec.name]
    assert isinstance(config, TrainConfig)
    assert config.total_frames >= 1
