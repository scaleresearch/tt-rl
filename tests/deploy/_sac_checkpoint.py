"""A SAC-shaped checkpoint built on the host, with no device and no agent.

`SAC.state_dict()` needs a chip to exist. What the exporter reads out of it is
a flat dict of numpy arrays under known names, so these build one directly and
every refusal and every numeric can be pinned by `pytest -m "not tt"`. The
hardware gate checks the real agent's state has these names
(`tests/hardware/test_deploy_parity.py`).
"""

from __future__ import annotations

import numpy as np

from ttrl.spaces import DEFAULT_LEAF

from ttrl.algos.sac.config import SACConfig
from ttrl.spaces import ActionSpec, ObsSpec
from ttrl.train.checkpoint import L0, save_checkpoint

OBS_DIM = 3
HIDDEN = (8, 6)
ACTION_DIM = 2

OBS_SPEC = ObsSpec.vector(OBS_DIM)
ACTION_SPEC = ActionSpec.continuous([-12.0, -4.0], [12.0, 4.0])


def _linear(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal((1, shape[1])).astype(np.float32))


def sac_nets(seed: int = 0) -> dict:
    """The parameter names `SacNets.state_dict()` produces, with real arrays.

    The critics are included because their encoders are what makes finding the
    actor's trunk by name a choice rather than a lookup. Every encoder is a
    `MultiEncoder`, so each tower's parameters carry their leaf's name.
    """
    rng = np.random.default_rng(seed)
    nets: dict[str, np.ndarray] = {}
    widths = (OBS_DIM,) + HIDDEN
    for layer, (fan_in, fan_out) in enumerate(zip(widths, HIDDEN)):
        weight, bias = _linear(rng, (fan_in, fan_out))
        # `Sequential` numbers ReLU too, so the Linears sit at 0, 2, 4...
        root = f"policy.encoder.towers.{DEFAULT_LEAF}.layers.{2 * layer}"
        nets[f"{root}.weight"] = weight
        nets[f"{root}.bias"] = bias
    for name in ("mean", "log_std"):
        weight, bias = _linear(rng, (HIDDEN[-1], ACTION_DIM))
        nets[f"policy.head.heads.{name}.weight"] = weight
        nets[f"policy.head.heads.{name}.bias"] = bias
    for group in ("critics", "targets"):
        for index in (0, 1):
            weight, bias = _linear(rng, (OBS_DIM + ACTION_DIM, 4))
            root = (f"{group}.{index}.encoder.encoder.towers."
                    f"{DEFAULT_LEAF}.layers.0")
            nets[f"{root}.weight"] = weight
            nets[f"{root}.bias"] = bias
    return nets


def agent_state(*, config=None, seed: int = 0) -> dict:
    return {
        "config": SACConfig() if config is None else config,
        "nets": sac_nets(seed),
        "log_alpha": np.full((1, 1), -1.0, np.float32),
        "weights_only": True,
    }


def write(path, *, env_policy_state, obs_spec=OBS_SPEC, action_spec=ACTION_SPEC,
          state=None, seed: int = 0) -> str:
    path = str(path)
    save_checkpoint(
        path,
        agent_state=agent_state(seed=seed) if state is None else state,
        obs_spec=obs_spec,
        action_spec=action_spec,
        level=L0,
        env_policy_state=env_policy_state,
    )
    return path
