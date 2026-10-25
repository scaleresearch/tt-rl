"""The recurrent actor-critic: trunk, then memory, then heads.

WHY THIS IS NOT AN ENCODER. `ttrl.nets.registry` is emphatic that the tower is
the architecture seam and that adding an architecture must not mean editing a
library file. Recurrence cannot ride that seam, and the reason is worth
stating here, where the next reader will be tempted to write a recurrent
encoder and register it. A tower's contract is stateless, `obs -> features`.
A recurrent trunk needs `(sequence, h0, starts) -> outputs` on the training
path, because the unroll's T is what fixes the captured graph, and it needs a
caller-owned carry on the acting path, because the state is both an input and
an output of one step. An encoder that satisfied `obs -> features` while
carrying memory would have to hide its state inside itself, and it would then
be hiding the three things this design is careful about: that the stored
initial state is stale by one epoch, that a reset can fall in the middle of a
sequence, and that a traced step must not read and write one aliased buffer.
So recurrence is a field on the algorithm's config and a different
composition here, and `ENCODERS` keeps meaning exactly what it says.

TWO MEMORIES WHEN THE TRUNKS DIFFER. An asymmetric observation
(`ObsGroup.actor` narrower than the group) gives the actor and the critic
separate trunks over different observations. Their memories are then separate
too, because one GRU over both would carry the critic's privileged features
into the state the actor reads, which is exactly the leak the spec's
actor/critic split exists to prevent, and it would be invisible: the run
trains and the policy is simply unusable on the deployed system. When the
group leaves every leaf to the actor the two trunks are one object and so is
the memory, which is the shared-trunk architecture.
"""

from __future__ import annotations

from ttrl.engine.nn import Module, ModuleDict
from ttrl.engine.ops import apply_done_mask, reshape
from ttrl.engine.recurrent import GRU
from ttrl.nets.heads import ValueHead, build_policy_head
from ttrl.nets.multi import build_encoder
from ttrl.nets.obs import ObsPair
from ttrl.spaces import ActionSpec, ObsGroup

__all__ = ["ACTOR", "CRITIC", "RecurrentActorCritic",
           "build_recurrent_actor_critic"]

#: The memory names. A buffer's stored hidden states, an agent's carries and
#: a minibatch's `h0` are all keyed by these, so they are defined once.
ACTOR = "actor"
CRITIC = "critic"


class RecurrentActorCritic(Module):
    """Trunk -> GRU -> policy and value heads, with the state supplied.

    The heads are sized from the GRU's hidden width rather than the trunk's,
    because the memory sits between them.

    `forward` is the training path and takes a whole minibatch of sequences:
    rows laid out time-major (`ttrl.buffers.sequences.flatten_time_major`),
    one `h0` per memory, and the `(seq_len, n_seq, 1)` episode-start mask the
    unroll applies to the state entering each timestep.

    `step` is the acting path and runs one timestep from a caller-owned
    `ttrl.engine.carry.ActingCarry`. It applies the SAME `apply_done_mask` to
    the state entering the cell that the unroll applies, so the two paths
    share one placement of one operator.
    """

    def __init__(self, actor, critic, memories: ModuleDict, policy: Module,
                 value: Module, hidden_size: int):
        self.encoders = ModuleDict({ACTOR: actor, CRITIC: critic})
        self.memories = memories
        self.heads = ModuleDict({"policy": policy, "value": value})
        self.hidden_size = int(hidden_size)
        self._shared = memories[ACTOR] is memories[CRITIC]

    @property
    def shared(self) -> bool:
        """Whether one trunk and one memory serve both heads.

        Stated as a flag at construction rather than read off object identity.
        Identity is what the builder uses to say it, and
        `ttrl.nets.acting.net_at_batch` preserves it, but a view of this net
        reporting a different number of memories than the net it views would
        make the acting carries and the rollout's stored states disagree about
        how many there are, and only at acting width.
        """
        return self._shared

    @property
    def hidden_sizes(self) -> dict:
        """Memory name -> hidden width, which is what sizes the rollout
        buffer's stored states and the acting carries. One entry when the
        memory is shared, so nothing stores a second copy of one state."""
        if self.shared:
            return {ACTOR: self.hidden_size}
        return {ACTOR: self.hidden_size, CRITIC: self.hidden_size}

    # -- training -----------------------------------------------------------

    def forward(self, obs: ObsPair, h0: dict, starts, *, seq_len: int):
        self._check_states(h0, "forward")
        actor_state = self._unroll(ACTOR, obs.actor, h0[ACTOR], starts, seq_len)
        policy_out = self.heads["policy"](actor_state)
        if self.shared:
            return policy_out, self.heads["value"](actor_state)
        critic_state = self._unroll(CRITIC, obs.critic, h0[CRITIC], starts,
                                    seq_len)
        return policy_out, self.heads["value"](critic_state)

    def _unroll(self, which: str, half: dict, h0, starts, seq_len: int):
        features = self.encoders[which](half)
        rows, width = (int(d) for d in features.shape)
        if rows % seq_len:
            raise ValueError(
                f"RecurrentActorCritic: {rows} rows is not a whole number of "
                f"sequences of seq_len={seq_len}. A recurrent minibatch is a set "
                f"of whole sequences, so choose a minibatch_size that is a "
                f"multiple of recurrent.seq_len."
            )
        n_seq = rows // seq_len
        sequence = reshape(features, (seq_len, n_seq, width))
        outputs, _final = self.memories[which](sequence, h0, starts)
        # Back to rows in the same time-major order, so every per-row term of
        # the loss indexes the batch the way a feedforward update does.
        return reshape(outputs, (rows, self.hidden_size))

    # -- acting -------------------------------------------------------------

    def step(self, obs: ObsPair, h_prev: dict, start: dict):
        """One timestep. Returns `(policy_out, value, h_next)`.

        `h_prev` and `start` are each keyed by memory name and hold the
        carry's `prev` and `start` buffers. `h_next` is what the caller binds
        into the carry, so the copy back happens outside any captured region.
        """
        self._check_states(h_prev, "step")
        self._check_states(start, "step")
        actor_next = self._cell(ACTOR, obs.actor, h_prev[ACTOR], start[ACTOR])
        policy_out = self.heads["policy"](actor_next)
        if self.shared:
            return policy_out, self.heads["value"](actor_next), {ACTOR: actor_next}
        critic_next = self._cell(CRITIC, obs.critic, h_prev[CRITIC],
                                 start[CRITIC])
        return (policy_out, self.heads["value"](critic_next),
                {ACTOR: actor_next, CRITIC: critic_next})

    def _cell(self, which: str, half: dict, h_prev, start):
        features = self.encoders[which](half)
        # The reset, in the one position the unroll also uses: on the state
        # ENTERING the cell. Masking what leaves it would zero this step's
        # output and cut its own observation out of the computation.
        return self.memories[which].cell(features, apply_done_mask(h_prev, start))

    def _check_states(self, states: dict, where: str) -> None:
        wanted = set(self.hidden_sizes)
        if set(states) != wanted:
            raise ValueError(
                f"RecurrentActorCritic.{where}: this network carries memories "
                f"{sorted(wanted)} and was given {sorted(states)}. An "
                f"asymmetric observation gives the actor and the critic "
                f"separate memories over different observations, so one state "
                f"cannot stand for both."
            )


def build_recurrent_actor_critic(obs_spec: ObsGroup, action_spec: ActionSpec,
                                 device, *, batch: int, hidden_size: int,
                                 encoder: str = "auto", head_kwargs=None,
                                 **encoder_kwargs) -> RecurrentActorCritic:
    """`build_actor_critic`'s recurrent twin: a GRU between trunk and heads.

    The actor's trunk is built over `obs_spec.actor` and the critic's over the
    whole group, exactly as in the feedforward case. When those are the same
    spec, one trunk and one GRU go into both slots; when they differ, the
    critic gets its own GRU. See TWO MEMORIES WHEN THE TRUNKS DIFFER above.
    """
    if hidden_size < 1:
        raise ValueError(
            f"build_recurrent_actor_critic: hidden_size must be >= 1, got "
            f"{hidden_size}"
        )
    actor = build_encoder(obs_spec.actor, device, batch=batch, name=encoder,
                          **encoder_kwargs)
    symmetric = obs_spec.actor is obs_spec
    critic = (actor if symmetric
              else build_encoder(obs_spec, device, batch=batch, name=encoder,
                                 **encoder_kwargs))
    actor_memory = GRU(actor.out_features, hidden_size, device)
    critic_memory = (actor_memory if symmetric
                     else GRU(critic.out_features, hidden_size, device))
    return RecurrentActorCritic(
        actor, critic,
        ModuleDict({ACTOR: actor_memory, CRITIC: critic_memory}),
        build_policy_head(action_spec, hidden_size, device,
                          **(head_kwargs or {})),
        ValueHead(hidden_size, device),
        hidden_size,
    )
