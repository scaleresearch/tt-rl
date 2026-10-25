"""The `import ttrl` surface: every advertised name resolves, and nothing else.

`ttrl/__init__.py` exports lazily through `__getattr__`, so a name can sit in
`__all__` with no branch behind it (`AttributeError` on first use) or have a
branch nobody advertises. Neither is caught by importing the package, only by
actually resolving every name, which is what this does.

What `import ttrl` must not drag in, and what it may cost, is pinned next
door in `tests/test_import_isolation.py`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

import ttrl

# The full user-facing story, in the order a user meets it: describe the
# problem (specs and the observation they describe), configure a run, train
# it, then score what it wrote.
EXPECTED = {
    "ObsField",
    "ObsSpec",
    "ObsGroup",
    "ObsLeaf",
    "Observation",
    "DEFAULT_LEAF",
    "ActionSpec",
    "mask_rows",
    "group_mask_rows",
    "draw_legal",
    "check_legal",
    "Agent",
    "AlgoConfig",
    "Transition",
    "check_agent",
    "check_algo_config",
    "ALGOS",
    "ENCODERS",
    "ENVS",
    "SAMPLERS",
    "ReplaySampler",
    "check_sampler",
    "UpdateCadence",
    "StepSource",
    "check_cadence",
    "EpisodeBoundaries",
    "check_boundaries",
    "check_env",
    "check_env_contract",
    "check_vec_env",
    "build_env",
    "RecordEpisodeStatistics",
    "DQNConfig",
    "PPOConfig",
    "RecurrentConfig",
    "PriorityConfig",
    "SACConfig",
    "TrainConfig",
    "DeviceConfig",
    "EnvConfig",
    "load_config",
    "device_for",
    "divide_for_replicas",
    "Trainer",
    "run",
    "evaluate",
    "evaluate_checkpoint",
    "record_video",
    "register_checkpoint_type",
    "config_from_checkpoint",
    "agent_from_checkpoint",
    "export_policy",
    "export_onnx",
    "load_policy",
    "Logger",
    "NullLogger",
    "JsonlLogger",
    "StdoutLogger",
    "MultiLogger",
    "build_logger",
    "TensorBoardLogger",
    "WandbLogger",
    "TrainHooks",
    "RunState",
    "StopOnRewardThreshold",
    "StopOnNoImprovement",
    "AnyOf",
    "algos",
    "deploy",
    "engine",
    "envs",
    "nets",
    "spaces",
    "train",
    "__version__",
}

#: The package directories that ship, globbed. A subpackage added without a
#: line here is a subpackage nothing says anything about either way, which is
#: how `ttrl.cli` came to be neither advertised nor declared unadvertised.
ON_DISK = tuple(sorted(
    path.name for path in (pathlib.Path(ttrl.__file__).parent).iterdir()
    if (path / "__init__.py").exists()
))

#: The ones deliberately NOT reachable as `ttrl.<name>`, each for a stated
#: reason. `buffers`: an agent owns its own storage, so a replay buffer is an
#: implementation detail. `cli`: the `ttrl` command is a shell over the public
#: surface and adds no capability, so it is reached as a console script rather
#: than composed into a run.
UNADVERTISED = ("buffers", "cli")

#: Every subpackage a user composes a run out of, reachable by name. `engine`
#: alone was, so `ttrl.envs` raised AttributeError until something else had
#: imported it, which made the whole surface depend on import order.
SUBPACKAGES = tuple(n for n in ON_DISK if n not in UNADVERTISED)

#: Names that moved into `ttrl.engine` and must not come back as aliases. A
#: shim here would let a caller keep the old spelling, which is exactly what
#: makes the engine's surface impossible to move afterwards.
MOVED_TO_ENGINE = ("device", "Tensor", "acquire", "Adam", "TrainStep", "no_grad")

def test_all_is_the_expected_surface():
    assert set(ttrl.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_resolves(name):
    """A name in `__all__` with no `__getattr__` branch raises here, not in a
    user's script three weeks after the release."""
    assert getattr(ttrl, name) is not None


def test_dir_covers_all():
    assert EXPECTED <= set(dir(ttrl))


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        ttrl.definitely_not_exported


@pytest.mark.parametrize("name", MOVED_TO_ENGINE)
def test_engine_names_are_not_re_exported_at_the_top_level(name):
    assert name not in ttrl.__all__
    with pytest.raises(AttributeError):
        getattr(ttrl, name)


def test_every_shipped_subpackage_is_advertised_or_declared_not_to_be():
    """The glob is the honest source: a new subpackage reaches this test on the
    day it is added rather than on the day someone edits a tuple."""
    assert len(ON_DISK) >= 7, ON_DISK
    assert set(UNADVERTISED) <= set(ON_DISK), (
        f"UNADVERTISED names {sorted(set(UNADVERTISED) - set(ON_DISK))}, which "
        f"are not packages; the ones that ship are {list(ON_DISK)}"
    )


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_each_subpackage_is_an_attribute(name):
    module = getattr(ttrl, name)
    assert module.__name__ == f"ttrl.{name}"


@pytest.mark.parametrize("name", UNADVERTISED)
def test_an_unadvertised_subpackage_is_not_an_attribute(name):
    """The other direction, so the exclusions above are a decision rather than
    a list nothing checks. A shim added later would make the surface larger
    than `__all__` says it is.

    In a subprocess, and that is the whole test rather than a detail:
    importing `ttrl.buffers` anywhere binds it as an attribute of `ttrl`, so
    this question is only answerable in an interpreter that has imported
    nothing else. Asked in process, it answered whichever way the rest of the
    suite happened to run.
    """
    assert name not in ttrl.__all__
    done = subprocess.run(
        [sys.executable, "-c",
         f"import ttrl; print(hasattr(ttrl, {name!r}))"],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", (
        f"ttrl.{name} resolves, but it is listed as unadvertised. Either "
        f"advertise it in __all__ and ttrl.__getattr__, or take the shim away."
    )


def test_a_logger_can_be_built_from_the_top_level(tmp_path):
    """A run takes a `Logger`; getting one must not need a private module path."""
    logger = ttrl.JsonlLogger(str(tmp_path))
    logger.log({"frame": 1}, 1)
    logger.close()
    assert isinstance(logger, ttrl.Logger)
