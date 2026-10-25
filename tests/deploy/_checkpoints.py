"""PPO's and DQN's agent states, built on the host with no device and no agent.

The SAC half of this is `tests/deploy/_sac_checkpoint.py`, which says why
these exist at all: the exporter reads a flat dict of numpy arrays under known
names, so building one directly puts every refusal and every numeric inside
`pytest -m "not tt"`. That the real agents write exactly these names is the
hardware gate's job (`tests/hardware/test_deploy_parity.py`).

Each state below carries the parts the exporter must NOT take: PPO's critic
trunk and value head, DQN's lagged target copy. An exporter that found the
actor by shape rather than by name would pick one of them up here.
"""

from __future__ import annotations

import numpy as np

from tests.deploy._sac_checkpoint import write  # noqa: F401  (re-exported)
from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.ppo.config import PPOConfig
from ttrl.nets.towers import NATURE_CNN
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsSpec

OBS_DIM = 3
HIDDEN = (8, 6)
ACTION_DIM = 2
N_ACTIONS = 5

OBS_SPEC = ObsSpec.vector(OBS_DIM)
DISCRETE = ActionSpec.discrete(N_ACTIONS)
CONTINUOUS = ActionSpec.continuous([-12.0, -4.0], [12.0, 4.0])

#: 44 -> 10 -> 4 -> 2 under the Nature CNN's strides. The last map must be
#: wider than one pixel, or NCHW and NHWC flatten to the same row.
IMAGE_SPEC = ObsSpec.image(2, 44, 44)


def linear(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal((1, shape[1])).astype(np.float32))


def tower(rng, nets: dict, prefix: str, in_features: int, hidden=HIDDEN) -> int:
    """One `MultiEncoder` of one leaf, `Sequential`-numbered as the real one is.

    `Sequential` numbers its ReLUs too, so the `Linear`s sit at 0, 2, 4...

    `hidden` is a parameter because one gate needs a SQUARE hidden layer: a
    fault injection that transposes a weight only measures the numerics when
    the transpose still fits, and with rectangular widths the runtime rejects
    it on shape before computing anything.
    """
    for layer, (fan_in, fan_out) in enumerate(zip((in_features,) + hidden, hidden)):
        weight, bias = linear(rng, (fan_in, fan_out))
        root = f"{prefix}towers.{DEFAULT_LEAF}.layers.{2 * layer}"
        nets[f"{root}.weight"] = weight
        nets[f"{root}.bias"] = bias
    return hidden[-1]


def ppo_nets(action_spec: ActionSpec, seed: int = 0, hidden=HIDDEN) -> dict:
    """What `ActorCritic.state_dict()` produces for one vector leaf.

    The critic trunk is present and differs from the actor's, which is the
    asymmetric case: in a symmetric run `ttrl.nets.build.build_actor_critic`
    puts ONE trunk in both slots and the two names hold the same arrays. The
    exporter must read `encoders.actor` either way.
    """
    rng = np.random.default_rng(seed)
    nets: dict[str, np.ndarray] = {}
    features = tower(rng, nets, "encoders.actor.", OBS_DIM, hidden)
    tower(rng, nets, "encoders.critic.", OBS_DIM, hidden)
    if action_spec.kind == "discrete":
        weight, bias = linear(rng, (features, action_spec.width))
        nets["heads.policy.logits.weight"] = weight
        nets["heads.policy.logits.bias"] = bias
    else:
        for name in ("mean", "log_std"):
            weight, bias = linear(rng, (features, action_spec.width))
            nets[f"heads.policy.heads.{name}.weight"] = weight
            nets[f"heads.policy.heads.{name}.bias"] = bias
    weight, bias = linear(rng, (features, 1))
    nets["heads.value.value.weight"] = weight
    nets["heads.value.value.bias"] = bias
    return nets


def ppo_state(action_spec: ActionSpec = DISCRETE, *, config=None,
              seed: int = 0, hidden=HIDDEN) -> dict:
    return {
        "config": PPOConfig() if config is None else config,
        "net": ppo_nets(action_spec, seed, hidden),
        "weights_only": True,
    }


def dqn_nets(action_spec: ActionSpec, seed: int = 0, hidden=HIDDEN,
             obs_dim: int = OBS_DIM) -> dict:
    """What `EncoderHead.state_dict()` produces for DQN's online Q network.

    `obs_dim` is a parameter because a legal-action mask occupies the tail of
    the observation, so a masked fixture needs a vector wider than its mask.
    """
    rng = np.random.default_rng(seed)
    nets: dict[str, np.ndarray] = {}
    features = tower(rng, nets, "encoder.", obs_dim, hidden)
    weight, bias = linear(rng, (features, action_spec.width))
    nets["head.q.weight"] = weight
    nets["head.q.bias"] = bias
    return nets


def dqn_state(action_spec: ActionSpec = DISCRETE, *, config=None,
              seed: int = 0, hidden=HIDDEN, obs_dim: int = OBS_DIM) -> dict:
    """Online and target both present: the exporter reads the online net."""
    return {
        "config": DQNConfig() if config is None else config,
        "online": dqn_nets(action_spec, seed, hidden, obs_dim),
        "target": dqn_nets(action_spec, seed + 100, hidden, obs_dim),
        "weights_only": True,
    }


def dqn_image_state(action_spec: ActionSpec = DISCRETE, *, seed: int = 0) -> dict:
    """DQN over `IMAGE_SPEC`, named as `image_encoder` names its layers.

    The conv weights are logical OIHW, as `Conv2d.state_dict` exports them,
    with biases in the device's `(1, 1, 1, C)` shape.
    """
    rng = np.random.default_rng(seed)
    nets: dict[str, np.ndarray] = {}
    channels = IMAGE_SPEC.shape[0]
    root = f"encoder.towers.{DEFAULT_LEAF}.layers"
    for index, stage in enumerate(NATURE_CNN):
        shape = (stage.out_channels, channels, stage.kernel, stage.kernel)
        fan_in = channels * stage.kernel ** 2
        nets[f"{root}.{2 * index}.weight"] = (
            rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)
        nets[f"{root}.{2 * index}.bias"] = rng.standard_normal(
            (1, 1, 1, stage.out_channels)).astype(np.float32) * 0.1
        channels = stage.out_channels
    after = 2 * len(NATURE_CNN) + 1
    nets[f"{root}.{after}.weight"], nets[f"{root}.{after}.bias"] = linear(
        rng, (channels * 2 * 2, HIDDEN[0]))
    nets["head.q.weight"], nets["head.q.bias"] = linear(
        rng, (HIDDEN[0], action_spec.width))
    return {"config": DQNConfig(), "online": nets, "weights_only": True}
