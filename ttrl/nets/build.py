"""The dispatcher: `(ObsSpec, ActionSpec) -> network`.

`obs_spec` is an `ObsGroup`: the whole observation, as the named leaves it is
made of. One encoder per leaf is `ttrl.nets.multi.build_encoder`'s job, and
which head to put on top is this module's, read off the `ActionSpec` and never
off which benchmark a spec came from. An unsupported combination raises here,
naming what is unsupported and what to call instead; nothing silently picks a
different architecture.

It is also the only module that decides which observation each network is
built over. An actor is built from `ObsGroup.actor`, so it has no tower
for a leaf the group marks critic-only and no first-layer weights for a field
a leaf marks critic-only; a critic is built from the group itself. In the symmetric case the two are the same spec, and
`build_actor_critic` then puts one trunk in both of its slots.
"""

from __future__ import annotations

from ttrl.engine.layers import Linear, ReLU, Sequential
from ttrl.engine.nn import ModuleList
from ttrl.nets.fusion import ConcatEncoder
from ttrl.nets.multi import build_encoder
from ttrl.nets.heads import GaussianHead, QHead, ValueHead, build_policy_head
from ttrl.nets.models import ActorCritic, ContinuousCritic, EncoderHead, SacNets
from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["build_actor_critic", "build_q_net", "build_sac_nets"]


def require_actor(obs_group: ObsGroup, where: str) -> ObsGroup:
    """`obs_group`, or a refusal naming what the actor must not read.

    For a component whose single network both acts and learns, which has
    nowhere to put a critic-only leaf or field. The group-level twin of
    `ObsSpec.require_actor`.
    """
    if obs_group.actor is not obs_group:
        raise ValueError(
            f"{where} builds one network that both chooses actions and is "
            f"trained, so {list(obs_group.privileged)} marked critic-only would "
            f"reach the actor. Use an algorithm with a separate critic ('ppo' or "
            f"'sac'), or drop the privileged leaves and fields from the "
            f"environment's observation spec."
        )
    return obs_group


def _require_discrete(action_spec: ActionSpec, what: str) -> None:
    if action_spec.kind == "multi_discrete":
        joint = 1
        for size in action_spec.nvec:
            joint *= int(size)
        raise ValueError(
            f"{what} supports plain discrete actions only, got a multi_discrete "
            f"ActionSpec with nvec={action_spec.nvec}. A value-based head needs one "
            f"output per action, and this action set holds {joint} joint actions. "
            f"Use build_actor_critic, whose policy head factorizes over the "
            f"factors, or enumerate the joint actions as "
            f"ActionSpec.discrete({joint})."
        )
    if action_spec.kind != "discrete":
        raise ValueError(
            f"{what} supports discrete actions only, got a continuous ActionSpec of "
            f"dim={action_spec.dim}. A value-based head would need one output per "
            f"action and a continuous action space has no finite action set. Use "
            f"build_sac_nets for continuous control."
        )


def _require_continuous(action_spec: ActionSpec, what: str) -> None:
    if action_spec.kind != "continuous":
        described = (f"multi_discrete ActionSpec with nvec={action_spec.nvec}"
                     if action_spec.kind == "multi_discrete"
                     else f"discrete ActionSpec with n={action_spec.n}")
        raise ValueError(
            f"{what} supports continuous actions only, got a {described}. Use "
            f"build_q_net for value-based discrete control or build_actor_critic "
            f"for a categorical policy."
        )


def build_q_net(obs_spec: ObsGroup, action_spec: ActionSpec, device, *, batch: int,
                encoder: str = "auto", **encoder_kwargs) -> EncoderHead:
    """Encoder + raw Q head, for DQN. Discrete actions only.

    One network that both chooses actions and is trained, so a spec with a
    critic-only field is refused by name: there is no critic here to put it in.
    """
    _require_discrete(action_spec, "build_q_net")
    tower = build_encoder(require_actor(obs_spec, "build_q_net"), device,
                          batch=batch, name=encoder, **encoder_kwargs)
    return EncoderHead(tower, QHead(tower.out_features, action_spec.width, device))


def build_actor_critic(obs_spec: ObsGroup, action_spec: ActionSpec, device, *, batch: int,
                       encoder: str = "auto", head_kwargs=None,
                       **encoder_kwargs) -> ActorCritic:
    """Actor trunk + critic trunk + policy head + value head, for PPO or A2C.

    Works for every action kind: `build_policy_head` picks categorical logits,
    one segment of logits per factor, or a Gaussian from `ActionSpec.kind`, and
    the value head is a scalar in all three cases.

    The actor trunk is built over `obs_spec.actor` and the critic trunk over
    the whole observation. A spec that leaves every field to the actor makes
    those the same network, and one trunk then goes into both slots, which is
    the shared trunk this architecture is usually written with.
    """
    actor = build_encoder(obs_spec.actor, device, batch=batch, name=encoder,
                          **encoder_kwargs)
    critic = (build_encoder(obs_spec, device, batch=batch, name=encoder,
                            **encoder_kwargs)
              if obs_spec.actor is not obs_spec else actor)
    policy = build_policy_head(action_spec, actor.out_features, device,
                               **(head_kwargs or {}))
    return ActorCritic(actor, critic, policy,
                       ValueHead(critic.out_features, device))


def build_sac_nets(obs_spec: ObsGroup, action_spec: ActionSpec, device, *, batch: int,
                   encoder: str = "auto", critic_encoder: str = "auto",
                   n_critics: int = 2,
                   critic_hidden=(256, 256), head_kwargs=None,
                   encoder_kwargs=None, critic_encoder_kwargs=None) -> SacNets:
    """Gaussian policy plus `n_critics` twin Q(s, a) critics and their targets.

    Continuous actions only. The critics are built over the whole observation
    and the policy over `obs_spec.actor`, which is what makes an asymmetric
    SAC (a critic that sees privileged simulator state, a policy that sees
    only what the real system can sense) the general case here.

    Each critic gets its own encoder: sharing one between the twins would
    correlate exactly the estimation errors the twin minimum exists to
    decorrelate.

    `encoder` names the POLICY's tower and `critic_encoder` the tower each
    critic runs before its action is concatenated on, which is what makes the
    fusion point of Q a declared choice. Both take anything
    `ttrl.nets.choice.resolve_encoder` does, and both default to 'auto'.

    'auto' is resolved per leaf kind AND per role: the critic's tower is built
    with `role="critic"`, so a vector leaf resolves to 'identity', a tower of
    no layers, which makes `concat(s, a)` the critic's input and `critic_hidden`
    the whole of Q, every layer of it depending on the action the actor
    differentiates. That is the published SAC critic (Haarnoja et al. 2018).
    An image leaf resolves to its own conv tower in both roles and fuses the
    action after it, because an action scalar has no spatial extent to occupy
    in a plane of pixels; 'identity' refuses an image spec by name rather than
    reading the shape and picking a fusion nobody asked for. The tables are
    `ttrl.nets.multi.DEFAULT_ENCODER_BY_ROLE`.

    The two towers take their builder keyword arguments separately, through
    `encoder_kwargs` and `critic_encoder_kwargs`, because they are two
    independent choices: 'identity' takes no `hidden`, so one shared mapping
    made sizing the policy's tower and fusing on the critic's input mutually
    exclusive. Each mapping reaches one builder, and a key that builder does
    not take raises there.

    `critic_hidden` sizes the hidden layers between the state-action
    concatenation and the scalar Q output, and must hold at least one. A
    critic whose head is a single `Linear` makes Q affine in the action, so
    dQ/da is a state-independent constant; SAC's actor maximises exactly that
    derivative, which drives the policy to one corner of the action cube
    regardless of the state. Measured by regression onto
    `Q(s, a) = -||a - tanh(s)||^2` (400 Adam steps, batch 256, target variance
    1.351): mse 0.40785 with a single Linear against 0.00369 with one hidden
    layer. 0.408 is the residual of the best action-affine fit, not slow
    convergence, so this is a representational limit and not a tuning choice.
    """
    _require_continuous(action_spec, "build_sac_nets")
    if n_critics < 2:
        raise ValueError(
            f"build_sac_nets: n_critics must be >= 2, got {n_critics}. SAC's target is "
            f"the minimum over the critics, which with one critic is that critic and "
            f"removes the overestimation correction entirely."
        )
    widths = tuple(int(w) for w in critic_hidden)
    if not widths or min(widths) < 1:
        raise ValueError(
            f"build_sac_nets: critic_hidden must hold at least one positive width, got "
            f"{critic_hidden}. With no hidden layer after the state-action concat, Q is "
            f"affine in the action and dQ/da is a state-independent constant, so the "
            f"actor maximising it walks to a corner of the action cube and the policy "
            f"cannot become state-dependent."
        )

    def one_critic() -> ContinuousCritic:
        tower = build_encoder(obs_spec, device, batch=batch, name=critic_encoder,
                              role="critic", **(critic_encoder_kwargs or {}))
        joint = ConcatEncoder(tower, action_spec.width)
        layers = []
        in_features = joint.out_features
        for width in widths:
            layers += [Linear(in_features, width, device), ReLU()]
            in_features = width
        # The scalar output stays a ValueHead so the critic's last layer is
        # named and shaped like every other value output in the package.
        layers.append(ValueHead(in_features, device))
        return ContinuousCritic(joint, Sequential(*layers))

    policy_tower = build_encoder(obs_spec.actor, device, batch=batch,
                                 name=encoder, **(encoder_kwargs or {}))
    policy = EncoderHead(
        policy_tower,
        GaussianHead(policy_tower.out_features, action_spec.width, device,
                     **(head_kwargs or {})),
    )

    nets = SacNets(
        policy=policy,
        critics=ModuleList([one_critic() for _ in range(n_critics)]),
        targets=ModuleList([one_critic() for _ in range(n_critics)]),
    )
    nets.sync_targets()
    return nets
