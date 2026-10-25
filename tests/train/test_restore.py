"""`agent_from_checkpoint`: specs come from the file, the loaded half of the state is decided, a foreign config is refused."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttrl.algos.checkpoint import assemble, validate
from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.registry import ALGOS
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec
from ttrl.train.checkpoint import L0, L1, read_meta, save_checkpoint
from ttrl.train.config import EnvConfig, TrainConfig
from ttrl.train.restore import agent_from_checkpoint

NAME = "restore-under-test"

#: The specs the FILE carries. Deliberately unlike anything the toy env or a
#: default config would produce, so a spec that came from the caller's config
#: instead of from the file cannot coincide with the right answer.
FILE_OBS = ObsSpec.vector(7)
FILE_ACTIONS = ActionSpec.discrete(5)

WEIGHTS = {"w": np.ones(3, dtype=np.float32)}
RUN = {"moment": np.zeros(3, dtype=np.float32), "frame": 11}


class RecordingAgent:
    """The `Agent` surface with the networks left out and the real loader in."""

    def __init__(self, obs_spec, action_spec, device, *, cfg=None, **kwargs):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.device = device
        self.cfg = cfg
        #: `(state, weights_only)` per `load_state_dict` call.
        self.loads: list[tuple[dict, bool]] = []

    def act(self, obs, *, deterministic=False, episode_starts=None):
        return np.zeros(int(obs.lanes), dtype=np.int64)

    def observe(self, transition): pass

    def update(self): return {}

    def should_update(self, frame): return False

    def named_parameters(self): return iter(())

    def parameters(self): return iter(())

    def state_dict(self, *, weights_only=False):
        return assemble(self.cfg, WEIGHTS, RUN, weights_only=weights_only)

    def load_state_dict(self, state, *, weights_only=False):
        validate(state, algo=NAME, config=self.cfg, weight_keys=tuple(WEIGHTS),
                 run_keys=tuple(RUN), weights_only=weights_only)
        self.loads.append((state, weights_only))

    def close(self): pass


@pytest.fixture(autouse=True)
def registered():
    ALGOS.register(NAME, RecordingAgent)
    try:
        yield
    finally:
        ALGOS.unregister(NAME)


def run_config(**overrides) -> TrainConfig:
    base = dict(algo=NAME, total_frames=64, log_dir="runs/restore",
                env=EnvConfig(name="toy", n_envs=1), algo_cfg=DQNConfig())
    return TrainConfig(**{**base, **overrides})


def write(path, *, level=L1, config=None, weights_only=False, algo_cfg=None):
    cfg = DQNConfig() if algo_cfg is None else algo_cfg
    save_checkpoint(
        str(path), level=level, frame=11,
        agent_state=assemble(cfg, WEIGHTS, RUN, weights_only=weights_only),
        obs_spec=FILE_OBS, action_spec=FILE_ACTIONS, env_policy_state=[],
        config=config if config is not None or level == L0 else run_config(),
    )
    return str(path)


DEVICE = object()


def test_the_agent_is_built_from_the_checkpoints_specs_whatever_config_it_is_given(
    tmp_path,
):
    path = write(tmp_path / "ckpt.npz")
    other = run_config(seed=3, env=EnvConfig(name="toy", n_envs=4))

    agent, cfg, loaded = agent_from_checkpoint(
        path, DEVICE, config=other, weights_only=True)
    assert agent.obs_spec == ObsGroup.of(FILE_OBS)
    assert agent.action_spec == FILE_ACTIONS
    assert agent.device is DEVICE
    assert cfg is other, "the caller's config is what it was asked to score under"
    assert loaded["frame"] == 11

    agent, cfg, _loaded = agent_from_checkpoint(path, DEVICE, weights_only=True)
    assert agent.obs_spec == ObsGroup.of(FILE_OBS)
    assert cfg.env.n_envs == 1 and cfg.seed == 0


def test_which_half_of_the_state_is_loaded_is_decided_and_never_inferred(tmp_path):
    full = write(tmp_path / "full.npz")

    agent, _cfg, _loaded = agent_from_checkpoint(full, DEVICE, weights_only=True)
    assert agent.loads[0][1] is True
    assert "frame" in agent.loads[0][0], "the run half is in the file, just unused"

    agent, _cfg, _loaded = agent_from_checkpoint(full, DEVICE)
    assert agent.loads[0][1] is False, "a plain call is a resume and takes both"

    snapshot = write(tmp_path / "snap.npz", level=L0, weights_only=True)
    agent, _cfg, _loaded = agent_from_checkpoint(
        snapshot, DEVICE, config=run_config())
    assert agent.loads[0][1] is True

    marked = write(tmp_path / "marked.npz", weights_only=True)
    with pytest.raises(ValueError) as caught:
        agent_from_checkpoint(marked, DEVICE)
    message = str(caught.value)
    assert "parameters only" in message
    assert "moment" in message and "frame" in message
    assert "weights_only=True" in message


def test_a_config_this_reader_had_to_complete_is_refused_even_for_weights(tmp_path):
    """A weights-only read is NOT config-free, which is what made this silent.

    `ttrl.train.checkpoint.tree` lets a field the file does not carry take
    today's default, on the ground that an optional setting cannot change the
    stored weight arrays. It cannot, and it is not what gets scored: the
    network is assembled from the stored algo config and a score's env is
    built from the stored env config, so an absent `normalize_obs`, obs clip,
    encoder or eval protocol scores a different function and labels the
    number this checkpoint's. A warm resume already refused these files;
    scoring accepted them.
    """

    path = write(tmp_path / "ckpt.npz")
    arrays, meta = read_meta(path)
    absent = meta["config"]["fields"].pop("eval_episodes")
    assert absent is not None, "the field must have been written to begin with"
    np.savez(path, **arrays, __meta__=np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8))

    for kwargs in ({"weights_only": True}, {}):
        with pytest.raises(ValueError) as caught:
            agent_from_checkpoint(path, DEVICE, **kwargs)
        message = str(caught.value)
        assert "TrainConfig.eval_episodes" in message, "it names the field"
        assert "network" in message and "environment" in message
        assert "Pass the config" in message, "and what the caller can do"

    # The named way out: given a config, nothing is read off the stored one,
    # so the caller owns every such choice and there is nothing to default.
    agent, cfg, _loaded = agent_from_checkpoint(
        path, DEVICE, config=run_config(), weights_only=True)
    assert agent.obs_spec == ObsGroup.of(FILE_OBS), "the specs still come from the file"
    assert cfg.eval_episodes == run_config().eval_episodes


def test_a_config_the_file_did_not_train_under_is_refused_by_name(tmp_path):
    path = write(tmp_path / "ckpt.npz", algo_cfg=DQNConfig(gamma=0.99))
    scored_as = run_config(algo_cfg=DQNConfig(gamma=0.5))

    with pytest.raises(ValueError) as caught:
        agent_from_checkpoint(path, DEVICE, config=scored_as, weights_only=True)
    message = str(caught.value)
    assert "configuration differs from the checkpoint's" in message
    assert "gamma=0.99" in message and "gamma=0.5" in message
    assert "n_step" not in message, "only the field that differs"
