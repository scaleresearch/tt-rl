"""Loading a :class:`TrainConfig` from a Python config module.

Usage::

    cfg = load_config("configs/pong_dqn.py")
    cfg = load_config("configs/pong_dqn.py", overrides=["algo_cfg.lr=5e-5"])
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from ttrl.envs import ENVS
from ttrl.train.config.coerce import _from_dict, config_to_dict
from ttrl.train.config.overrides import apply_overrides
from ttrl.train.config.schema import RUN_HORIZON_FIELD, TrainConfig


def _load_config_module(path: str):
    """Import the Python module at ``path`` (e.g. ``configs/pong_dqn.py``).

    Loaded by file path rather than as a dotted import, which keeps ``path`` a
    plain string: that is the shape a command line and a subprocess boundary
    both need.
    """
    file_path = Path(path)
    if file_path.suffix != ".py":
        raise ValueError(f"{path}: config path must be a .py file (plain Python config, no YAML)")
    if not file_path.is_file():
        raise FileNotFoundError(f"config module not found: {path}")

    # Unique per resolved path: two `configs/pong_dqn.py` under different roots
    # are different modules, and the name must not collide with a real import.
    resolved = file_path.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    mod_name = f"ttrl_config_{file_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered *before* execution, as a normal import would: a config module
    # that defines a dataclass or an enum (or anything else that looks itself
    # up by `__module__`) breaks without this.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[mod_name]
        # A config module is user code; an exception raised while executing it
        # otherwise surfaces as a bare traceback with no hint of which config
        # was being loaded. Chained, so the original traceback is intact.
        raise RuntimeError(f"failed to load config module {path}: {exc}") from exc
    return module


def load_config(
    path: str,
    overrides: dict[str, Any] | list[str] | None = None,
) -> TrainConfig:
    """Load a :class:`TrainConfig` from a Python config module, applying overrides.

    Args:
        path: path to a ``.py`` file exporting a module-level ``config``
            attribute -- a :class:`TrainConfig` built directly from the
            dataclasses in :mod:`.schema` (e.g. ``configs/pong_dqn.py``). A
            module missing ``config``, or exporting something that is not a
            :class:`TrainConfig`, fails loudly (``AttributeError`` /
            ``TypeError``) rather than silently falling back to a default.
        overrides: dotted-path overrides, either as a dict
            (``{"algo_cfg.lr": 5e-5}``) or a list of ``"algo_cfg.lr=5e-5"`` strings, the
            shape ``key=value`` command-line arguments naturally produce.
    """
    module = _load_config_module(path)
    if getattr(module, "config", None) is None:
        raise AttributeError(
            f"{path}: config module must define a module-level "
            "`config = TrainConfig(...)` (see configs/pong_dqn.py)"
        )
    config = module.config
    if not isinstance(config, TrainConfig):
        raise TypeError(f"{path}: module.config must be a TrainConfig, got {type(config)!r}")

    if not overrides:
        return _check_env_names(config)

    raw = _reopen_derived(config, config_to_dict(config))
    raw = apply_overrides(raw, overrides)
    # `algo_cfg` holds the algorithm's own dataclass, whose type this module
    # does not know and must not guess. The loaded config does know it, so the
    # overridden fields are rebuilt into that same type here.
    if isinstance(raw.get("algo_cfg"), dict):
        raw["algo_cfg"] = _from_dict(type(config.algo_cfg), raw["algo_cfg"])
    return _check_env_names(_from_dict(TrainConfig, raw))


def _reopen_derived(config: TrainConfig, raw: dict[str, Any]) -> dict[str, Any]:
    """Set every value the config DERIVED back to unset, before overrides.

    `TrainConfig.__post_init__` resolves each of these from the one source it
    belongs to, so the serialized dict carries the resolved number and looks
    as explicitly supplied as a number the user typed. Overriding the source
    would then leave the derived copy on its old value, or be refused for
    disagreeing with it. Both are the same bug: an override that applies to
    half of one quantity. Which values were derived is recorded at the
    derivation sites (`EnvConfig._seed_follows_run_seed`,
    `EnvConfig._reward_gamma_follows_algo`,
    `TrainConfig._run_horizon_follows_run`), so intent is read here rather
    than guessed, and only those are re-opened: a value the user supplied is
    carried through, and a disagreement with it is still refused by name.

    An explicit override of a derived field still wins; it is applied after
    this, and still checked against its source.
    """
    follows = (
        ("_seed_follows_run_seed", "seed"),
        ("_reward_gamma_follows_algo", "normalize_reward_gamma"),
    )
    for section in ("env", "eval_env"):
        env = getattr(config, section)
        if env is None or not isinstance(raw.get(section), dict):
            continue
        for marker, field_name in follows:
            if getattr(env, marker, False):
                raw[section][field_name] = None
    if config._run_horizon_follows_run and isinstance(raw.get("algo_cfg"), dict):
        raw["algo_cfg"][RUN_HORIZON_FIELD] = None
    return raw


def _check_env_names(config: TrainConfig) -> TrainConfig:
    """Refuse an unregistered environment name here, where it costs nothing.

    `open_replicas` opens a chip before it builds an environment, so an
    unregistered name used to cost a chip acquisition and about twenty
    seconds of ttnn initialisation before the registry said no. In a queued
    sweep every run pays that before failing.

    The env registry lists its names without importing any builder, so this
    stays free of ttnn and of every environment suite. `config.algo` is NOT
    checked here for the opposite reason: `ALGOS.names()` imports the
    algorithm modules and therefore ttnn, and a run has to set
    TT_VISIBLE_DEVICES before the first `import ttnn` or it takes the whole
    board. `ttrl.train.run` checks it immediately after scoping, which is
    still before any device is opened.
    """

    known = ENVS.names()
    for label, env in (("env", config.env), ("eval_env", config.eval_env)):
        if env is not None and env.name not in known:
            raise ValueError(
                f"{label}.name {env.name!r} is not a registered environment; "
                f"registered names are {known}. Fix the name, or register "
                "yours with ttrl.envs.register(name, builder) before "
                "loading the config."
            )
    return config
