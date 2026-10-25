"""Where each algorithm keeps its actor, and what its action map is.

ONE TABLE, read by the exporter and enumerated by the tests. Every algorithm
in `ttrl.algos.registry.ALGOS` has an entry here, so "every algorithm exports"
is a claim a test derives from the registry rather than from a list someone
typed: `tests/deploy/test_export.py` resolves `ALGOS.names()` and refuses a
registered name this table does not cover.

Finding the actor is done BY NAME in the checkpoint's flat parameter dict,
never by rebuilding the networks, which would need a chip. Each entry carries
the state member its weights live under, an anchored pattern matching only its
ACTOR tower's `Linear` weights, and the head the deterministic action is read
off. The anchoring is what keeps a critic out: SAC's state holds four encoders
in one flat dict and `^policy\\.encoder\\.` matches exactly one of them.

An algorithm's action map depends on its action kind, so each entry maps a
kind to a map name and an action kind absent from that mapping is refused by
name. The three maps and their arithmetic live in `ttrl.deploy.artifact`.

WHAT THE EXPORTED DQN POLICY IS. `DQNConfig.eval_epsilon` defaults to 0.05, so
`DQN.act(deterministic=True)` takes a uniformly random action one step in
twenty. That allowance belongs to evaluating an Atari agent whose training
autoresets supplied the start action for it, and the artifact deliberately
does not carry it: a deployed controller that takes a random action every
twentieth step is a hazard, and the greedy policy is what the Q values mean.
The exported file is `argmax Q`, and a parity check against the agent runs it
with `eval_epsilon=0`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.ppo.config import PPOConfig
from ttrl.algos.sac.config import SACConfig

__all__ = ["SOURCES", "PolicySource", "source_for"]


@dataclass(frozen=True)
class PolicySource:
    """Where one algorithm's actor is, and how its row becomes an action."""

    algo: str
    config: type
    #: The `agent_state` member holding this algorithm's flat parameter dict.
    state_key: str
    #: Matches the actor tower's `Linear` weights and nothing else. Groups:
    #: `leaf` (the observation leaf this tower reads) and `index` (the
    #: layer's position inside `Sequential`, which numbers the ReLUs too).
    trunk: re.Pattern = field(compare=False)
    #: Action kind -> the head weight's parameter name.
    heads: dict[str, str] = field(compare=False)
    #: Action kind -> the action map `ttrl.deploy.artifact` applies.
    maps: dict[str, str] = field(compare=False)
    #: Prefix of a recurrent memory's parameters, for an algorithm that can
    #: train one. The artifact is a feedforward map with no carried state.
    memory: str | None = field(default=None, compare=False)


def _tower(prefix: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(prefix)}towers\.(?P<leaf>[^.]+)\.layers\."
        rf"(?P<index>\d+)\.weight$")


SOURCES = (
    PolicySource(
        algo="sac",
        config=SACConfig,
        state_key="nets",
        trunk=_tower("policy.encoder."),
        heads={"continuous": "policy.head.heads.mean.weight"},
        maps={"continuous": "tanh_bounds"},
    ),
    PolicySource(
        algo="ppo",
        config=PPOConfig,
        state_key="net",
        trunk=_tower("encoders.actor."),
        heads={"continuous": "heads.policy.heads.mean.weight",
               "discrete": "heads.policy.logits.weight"},
        # PPO's continuous policy CLIPS its sample to the bounds where SAC's
        # squashes with tanh, so the same weights are a different function
        # and the two must not share a map.
        maps={"continuous": "clip_bounds", "discrete": "argmax"},
        memory="memories.",
    ),
    PolicySource(
        algo="dqn",
        config=DQNConfig,
        state_key="online",
        trunk=_tower("encoder."),
        heads={"discrete": "head.q.weight"},
        maps={"discrete": "argmax"},
    ),
)


def source_for(algo_config, path: str) -> PolicySource:
    """The entry for the config a checkpoint carries, or a refusal naming it."""
    for source in SOURCES:
        if type(algo_config) is source.config:
            return source
    known = ", ".join(f"{s.config.__name__} ({s.algo})" for s in SOURCES)
    raise ValueError(
        f"cannot export a deployable policy: {path!r} was trained with "
        f"{type(algo_config).__name__}, and this exporter knows {known}. An "
        f"algorithm's weights only mean a policy through the map that turns "
        f"its head's output into an action, so a config this file does not "
        f"name cannot be exported by guessing at one. Add an entry to "
        f"ttrl.deploy.policies."
    )
