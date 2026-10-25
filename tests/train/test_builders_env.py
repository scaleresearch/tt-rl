"""What `build_env` refuses, and what it has to carry into a spawned lane."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests._fakes.train_layer import ToyEnv, make_toy
from ttrl.envs import ENVS, register
from ttrl.envs.base import EPISODE_KEY, LANE_KEY
from ttrl.envs.wrappers import TimeLimit
from ttrl.train.builders import build_env, require_episode_bound
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.trainer import Trainer

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def unregistered():
    """Names to take back out of the process-global registry after the test."""
    names: list[str] = []
    yield names
    for name in names:
        ENVS.unregister(name)


def test_an_env_with_no_statistics_of_its_own_builds_and_is_scorable(unregistered):
    """The wrapper is not mandatory, and the score is still there."""
    register("bare_toy", lambda **kwargs: ToyEnv(**kwargs))
    unregistered.append("bare_toy")
    env = build_env(EnvConfig(name="bare_toy", n_envs=2))
    try:
        env.reset(seed=0)
        ended = []
        for _ in range(40):
            _o, _r, term, trunc, info = env.step(
                np.zeros(env.num_envs, dtype=np.int64))
            done = np.asarray(term, bool) | np.asarray(trunc, bool)
            ended += [info[LANE_KEY][lane][EPISODE_KEY]
                      for lane in np.flatnonzero(done).tolist()]
        assert ended, "no lane ended in 40 steps, so nothing was asserted"
        assert all(set(e) == {"r", "l"} and e["l"] >= 1 for e in ended)
    finally:
        env.close()


class Unending:
    """An `Env` whose episodes never end and which declares no bound."""

    obs_spec = ToyEnv().obs_spec
    action_spec = ToyEnv().action_spec

    def reset(self, *, seed=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

    def close(self):
        pass


def test_an_unbounded_eval_env_is_refused_at_run_start(unregistered, tmp_path):
    """Before the chip and the first frame, by name."""
    register("unending", lambda **kwargs: Unending())
    unregistered.append("unending")
    config = TrainConfig(
        algo="ppo",
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", n_envs=2, kwargs={"episode_len": 3}),
        eval_env=EnvConfig(name="unending", n_envs=2),
    )
    with pytest.raises(ValueError) as excinfo:
        Trainer(config)
    message = str(excinfo.value)
    assert "unending" in message and "eval_env" in message
    assert "TimeLimit" in message and "episode_bound" in message
    register("timed_toy", lambda **kwargs: TimeLimit(ToyEnv(episode_len=9), 2))
    unregistered.append("timed_toy")
    assert require_episode_bound(EnvConfig(name="timed_toy"), where="env") == 2


def test_a_builder_defined_in_a_function_is_refused_before_a_lane_is_spawned(
    unregistered,
):
    def make_local():
        return make_toy()

    register("local_toy", make_local)
    unregistered.append("local_toy")
    with pytest.raises(ValueError) as excinfo:
        build_env(EnvConfig(name="local_toy", n_envs=2, asynchronous=True))
    message = str(excinfo.value)
    assert "subprocess" in message
    assert "asynchronous" in message
    assert "local_toy" in message
    env = build_env(EnvConfig(name="local_toy", n_envs=2))
    env.close()


SCRIPT = '''
"""A user's entry point: the env is a module-level class, registered in main()."""
import numpy as np

import ttrl
from ttrl.envs.wrappers import RecordEpisodeStatistics


class GridWorld:
    obs_spec = ttrl.ObsSpec.vector(2)
    action_spec = ttrl.ActionSpec.discrete(4)

    def __init__(self):
        self._t = 0

    def reset(self, *, seed=None):
        self._t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        return (np.zeros(2, dtype=np.float32), 1.0, self._t >= 4, False, {})

    def close(self):
        pass


def make_gridworld():
    return RecordEpisodeStatistics(GridWorld())


def main():
    ttrl.envs.register("gridworld2", make_gridworld)
    from ttrl.train.builders import build_env

    env = build_env(ttrl.EnvConfig(name="gridworld2", n_envs=2, asynchronous=True))
    try:
        env.reset(seed=0)
        for _ in range(6):
            env.step(np.zeros(2, dtype=np.int64))
    finally:
        env.close()
    print("OK")


if __name__ == "__main__":
    main()
'''


def test_an_env_registered_in_main_works_with_spawned_lanes(tmp_path):
    """The obvious code: register inside `main()`, ask for async lanes."""
    script = tmp_path / "entry.py"
    script.write_text(textwrap.dedent(SCRIPT))
    done = subprocess.run([sys.executable, str(script)], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout
