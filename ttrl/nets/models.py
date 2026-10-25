"""The composed networks: encoder plus head(s).

Each is a thin `Module` over an encoder and heads that were already chosen
from the specs, so none of them inspects an `ObsSpec` or an `ActionSpec`
itself. That choosing happens once, in `ttrl.nets.build`.

Every one of them is called with an `ObsPair` and reads the half it was built
for: an actor reads `pair.actor`, which carries only the fields the `ObsSpec`
leaves to the actor, and a critic reads `pair.critic`, which carries the whole
observation. No call site picks a half, so privileged simulator state cannot
reach an actor by being passed to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ttrl.engine.nn import Module, ModuleDict, ModuleList
from ttrl.nets.encoders import Encoder
from ttrl.nets.fusion import ConcatEncoder
from ttrl.nets.obs import ObsPair

__all__ = ["ActorCritic", "ContinuousCritic", "EncoderHead", "SacNets"]


class EncoderHead(Module):
    """Encoder -> one head. `forward(obs)` returns whatever the head returns.

    One class serves DQN's Q network and SAC's Gaussian policy, because that
    is the same composition twice: which head was chosen is the specs' job and
    was already decided in `ttrl.nets.build`.
    """

    def __init__(self, encoder: Encoder, head: Module):
        self.encoder = encoder
        self.head = head

    def forward(self, obs: ObsPair):
        return self.head(self.encoder(obs.actor))


class ActorCritic(Module):
    """An actor trunk, a critic trunk, a policy head and a value head.

    The trunks and the heads are each a `ModuleDict`, so parameters are named
    `encoders.actor.*` and `heads.policy.*`.

    Asymmetric observation is the general case: the critic's trunk is built
    over the whole observation and the actor's over the actor's fields alone.
    When the `ObsSpec` leaves every field to the actor, `ttrl.nets.build`
    passes ONE trunk into both slots, which is the shared-trunk architecture:
    one encoder forward feeds both heads and the trunk's gradient is the sum
    of theirs.

    `forward(ObsPair) -> (policy_out, value)`. `policy_out` is logits of width
    `n` for discrete actions and `(mean, log_std)` for continuous ones, which
    is exactly what the chosen head returns.
    """

    def __init__(self, actor: Encoder, critic: Encoder, policy: Module,
                 value: Module):
        self.encoders = ModuleDict({"actor": actor, "critic": critic})
        self.heads = ModuleDict({"policy": policy, "value": value})

    def forward(self, obs: ObsPair):
        actor, critic = self.encoders["actor"], self.encoders["critic"]
        features = actor(obs.actor)
        # One trunk in both slots is the shared-trunk case, where the two
        # halves are the same columns, so it runs once.
        critic_features = features if critic is actor else critic(obs.critic)
        return self.heads["policy"](features), self.heads["value"](critic_features)


class ContinuousCritic(Module):
    """Q(s, a) for continuous actions: encode s, append a, then a value head.

    Reads the critic's half, which is the whole observation including any
    field the actor is not allowed to see.
    """

    def __init__(self, encoder: ConcatEncoder, head: Module):
        self.encoder = encoder
        self.head = head

    def forward(self, obs: ObsPair, action):
        return self.head(self.encoder(obs.critic, action))


@dataclass
class SacNets:
    """SAC's networks, built together because they must agree on their specs.

    `critics` and `targets` are `ModuleList`s of two: twin critics are an
    ordered homogeneous collection held by index, which is what `ModuleList`
    is for, and it gives them stable names (`0.encoder...`, `1.encoder...`) so
    `copy_from` can sync target `i` from critic `i` by name rather than by
    position.
    """

    policy: Module
    critics: ModuleList
    targets: ModuleList

    def sync_targets(self) -> None:
        """Hard-copy each critic into its own target, in place."""
        for critic, target in zip(self.critics, self.targets, strict=True):
            target.copy_from(critic)

    # No `parameters()` over the whole bundle, deliberately. The targets are
    # not trained, and a single all-parameters accessor is the one call an
    # optimizer would naturally be handed, which would silently start updating
    # them and quietly remove the delayed-target machinery SAC depends on.
    def policy_parameters(self):
        return list(self.policy.parameters())

    def critic_parameters(self):
        return list(self.critics.parameters())

    def state_dict(self) -> dict:
        """Targets included: a resume that rebuilt them from the critics would
        discard however far they had lagged, which is state, not derived data."""
        out = {}
        for name, module in (("policy", self.policy), ("critics", self.critics),
                             ("targets", self.targets)):
            for key, array in module.state_dict().items():
                out[f"{name}.{key}"] = array
        return out

    def load_state_dict(self, state: dict) -> None:
        for name, module in (("policy", self.policy), ("critics", self.critics),
                             ("targets", self.targets)):
            prefix = f"{name}."
            module.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            )
