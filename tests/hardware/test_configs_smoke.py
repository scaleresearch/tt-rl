"""The shipped `configs/*.py` actually run. "Add a game = one config module."

The real config modules are loaded, not hand-built TrainConfigs, their frame
budgets shrunk through `load_config`'s override mechanism, and each run
through the real `Trainer`. Not a training run: this proves the config loads,
that the registered env builder resolves its id with the right action space
and frame shape, and that a few real update steps execute.

Chip-scoped by the caller:

    TT_VISIBLE_DEVICES=0 pytest tests/hardware/test_configs_smoke.py -m tt
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from ttrl.envs.registry import make

pytest.importorskip("ttnn")

from ttrl.engine import acquire
from ttrl.envs.base import EPISODE_KEY, LANE_KEY
from ttrl.train.builders import build_env, require_episode_bound
from ttrl.train.config import load_config
from ttrl.train.evaluate import MAX_EPISODE_STEPS
from ttrl.train.trainer import Trainer

# `tt` is per test, not per module: `test_env_matches_config` builds an ALE
# env only and belongs on the CPU gate.

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

#: Every shipped game, derived from the config modules rather than typed. The
#: list that used to stand here sampled three games and said so, and the
#: sample is what hid a real defect: Breakout's evaluation environment runs
#: whole games with no life-loss reset, and the ball needs FIRE to come back
#: into play, so a deterministic policy held a ball-less game open until the
#: eval step guard fired. None of the three sampled games can stall that way.
#: Device time for the full set is seconds per config, so the set is the set.
GAMES = sorted(
    {p.name[: -len("_dqn.py")] for p in Path(CONFIG_DIR).glob("*_dqn.py")}
    & {p.name[: -len("_ppo.py")] for p in Path(CONFIG_DIR).glob("*_ppo.py")}
)

#: The action-space width each game's ALE id must resolve to. Keyed by ALE id,
#: which comes off the config, and checked against GAMES below, so adding a
#: config without stating its width fails rather than going unchecked.
ACTION_COUNTS = {
    "ALE/Boxing-v5": 18,
    "ALE/Breakout-v5": 4,
    "ALE/Pong-v5": 6,
    "ALE/SpaceInvaders-v5": 6,
}


#: Every shipped config module, so a config added for a new suite is covered
#: without being named here.
CONFIGS = sorted(
    p.name for p in Path(CONFIG_DIR).glob("*.py") if not p.name.startswith("_")
)


def _ale_id(game: str) -> str:
    return load_config(os.path.join(CONFIG_DIR, f"{game}_dqn.py")).env.kwargs["game"]


def test_every_shipped_game_is_covered():
    assert len(GAMES) >= 4, GAMES
    unstated = sorted(g for g in GAMES if _ale_id(g) not in ACTION_COUNTS)
    assert not unstated, f"no expected action count for {unstated}"


def _dqn_overrides(tmp_path, game: str) -> list[str]:
    return [
        f"log_dir={tmp_path / (game + '_dqn_run')}",
        "total_frames=200",
        "eval_interval=100",
        "eval_episodes=1",
        "checkpoint_interval=100",
        "algo_cfg.target_sync_interval=5",
        "algo_cfg.buffer_capacity=1000",
        "algo_cfg.eps_decay_frames=1000",
        "algo_cfg.batch_size=4",
        # A 200-frame smoke run: the shipped 80k warm-up would outlast it and
        # no update would happen. This is about the config wiring, not the
        # warm-up schedule.
        "algo_cfg.learning_starts=4",
    ]


def _ppo_overrides(tmp_path, game: str) -> list[str]:
    return [
        f"log_dir={tmp_path / (game + '_ppo_run')}",
        "total_frames=64",
        "eval_interval=32",
        "eval_episodes=1",
        "checkpoint_interval=32",
        "env.n_envs=2",
        "eval_env.n_envs=1",
        "algo_cfg.n_steps=8",
        "algo_cfg.epochs=1",
        "algo_cfg.minibatch_size=8",
        "algo_cfg.total_frames=64",
    ]


def _assert_ran(config, result):
    """Checked after the Trainer has closed: the JSONL sink is buffered, so a
    read taken while the run is still open sees whatever happens to have been
    flushed."""
    assert result["frame_count"] >= config.total_frames
    assert os.path.exists(os.path.join(config.log_dir, "metrics.jsonl"))
    with open(os.path.join(config.log_dir, "metrics.jsonl")) as fh:
        lines = [line for line in fh if line.strip()]
    assert any("loss" in line for line in lines), "no training metric was logged"


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.tt
def test_dqn_config_smoke(game, tmp_path):
    config = load_config(os.path.join(CONFIG_DIR, f"{game}_dqn.py"),
                         overrides=_dqn_overrides(tmp_path, game))
    assert config.algo == "dqn"
    with acquire(0, cfg=config.device.engine_config()) as (dev, profile):
        with Trainer(config, dev, profile) as trainer:
            result = trainer.run()
    _assert_ran(config, result)


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.tt
def test_ppo_config_smoke(game, tmp_path):
    config = load_config(os.path.join(CONFIG_DIR, f"{game}_ppo.py"),
                         overrides=_ppo_overrides(tmp_path, game))
    assert config.algo == "ppo"
    with acquire(0, cfg=config.device.engine_config()) as (dev, profile):
        with Trainer(config, dev, profile) as trainer:
            result = trainer.run()
    _assert_ran(config, result)


@pytest.mark.parametrize("game", GAMES)
def test_env_matches_config(game):
    """Each config's env id resolves through the registry and wraps correctly."""

    config = load_config(os.path.join(CONFIG_DIR, f"{game}_dqn.py"))
    assert config.env.name == "atari"
    ale_id = config.env.kwargs["game"]

    env = make(config.env.name, **config.env.kwargs, seed=0)
    try:
        assert env.action_spec.n == ACTION_COUNTS[ale_id]
        assert env.obs_spec.shape == (4, 84, 84)
        obs, _info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype.name == "uint8"
        obs, _r, _t, _tr, _i = env.step(0)
        assert obs.shape == (4, 84, 84)
    finally:
        env.close()


@pytest.mark.parametrize("game", GAMES)
def test_the_eval_env_is_the_ale_protocol(game):
    """The published number must come off whole-game episodes with untransformed
    rewards, whatever shaping training applies."""
    config = load_config(os.path.join(CONFIG_DIR, f"{game}_dqn.py"))
    eval_env = config.eval_env_config
    assert eval_env is not config.env
    assert eval_env.kwargs["episodic_life"] is False
    assert eval_env.kwargs["reward"] == "none"
    assert eval_env.kwargs["game"] == config.env.kwargs["game"]


@pytest.mark.parametrize("config_name", CONFIGS)
def test_every_shipped_eval_env_bounds_its_episodes_below_the_guard(config_name):
    """Two halves of one defect, held together.

    An eval environment whose episodes cannot end stalls the run that scores
    on it, because eval runs inline in the training loop. A guard BELOW the
    bound the environment installs is the other way to lose a run: it fires on
    a legitimately long episode the environment would have truncated and
    scored by itself, which is what killed a Breakout run six hours in at a
    guard of 10,000 agent steps against the ALE protocol's 108,000 frames.

    Asked of every config on disk rather than a sample. The sample is what hid
    the last defect of this shape.
    """
    config = load_config(os.path.join(CONFIG_DIR, config_name))
    bound = require_episode_bound(config.eval_env_config, where="eval_env")
    assert bound <= MAX_EPISODE_STEPS, (
        f"{config_name} scores on episodes of up to {bound} steps while the "
        f"eval guard refuses to wait past {MAX_EPISODE_STEPS}"
    )


@pytest.mark.parametrize("game", GAMES)
def test_the_eval_bound_truncates_into_a_scored_episode(game):
    """A bound is only worth anything if a truncation is an episode END.

    Scored the way `ttrl.train.evaluate` scores: a truncation must arrive as
    `truncated`, not as a termination, and must carry the lane's return where
    the evaluation reads it from. The frame limit is shortened so the bound
    arrives in a few hundred frames instead of 108,000; nothing else about
    the shipped eval config is changed.
    """
    config = load_config(os.path.join(CONFIG_DIR, f"{game}_ppo.py"))
    eval_env = config.eval_env_config
    cfg = dataclasses.replace(
        eval_env, n_envs=1, kwargs={**eval_env.kwargs, "max_frames": 400})
    env = build_env(cfg)
    try:
        env.reset(seed=0)
        for _ in range(400):
            _o, _r, term, trunc, info = env.step(np.zeros(1, dtype=np.int64))
            if term[0] or trunc[0]:
                break
        assert trunc[0] and not term[0], (
            f"{game}: the frame bound ended the episode as {term[0]=} "
            f"{trunc[0]=}, so it is not a truncation"
        )
        assert EPISODE_KEY in info[LANE_KEY][0], (
            f"{game}: a truncated episode carries no score where "
            f"ttrl.train.evaluate reads one"
        )
        assert info[LANE_KEY][0][EPISODE_KEY]["l"] == 400
    finally:
        env.close()
