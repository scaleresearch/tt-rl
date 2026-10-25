"""What PPO does differently when its policy has memory.

Three things, and they are all here so that `ttrl.algos.ppo.agent` reads the
same whether the policy remembers anything or not.

THE ACTING CARRY. One `ttrl.engine.carry.ActingCarry` per memory, per lane
width, per purpose. Per memory because an asymmetric observation gives the
actor and the critic separate memories. Per width because evaluation runs a
different lane count from training and decides it from its own `EnvConfig`,
which is the same reason `ttrl.nets.acting.ActingNet` keys its views on width.
Per purpose because a deterministic evaluation rollout run between two
training steps would otherwise overwrite the training carry with the
evaluation episode's hidden state, and the training rollout would resume from
a state belonging to a different episode of a different environment instance.
Nothing in the shapes would catch that.

WHERE THE EPISODE-START FLAGS COME FROM. During training, from the agent's
declared `ttrl.EpisodeBoundaries` (`ttrl.algos.ppo.boundaries`), read once per
`observe` and defaulting to the previous step's `terminated or truncated`, so
no caller supplies it and an ablation that wants another answer has a seam to
say so through rather than this object's privates. During evaluation PPO sees no transitions at all, only
`act(deterministic=True)`, so the flags must come from the caller and their
absence raises. A deterministic acting call that silently carried state across
an episode boundary would report an evaluation score for a policy nobody is
running.

THE LOSS REDUCES OVER VALID SLOTS. A sequence keeps its autoreset padding,
because dropping a slot shortens the sequence and so changes the graph shape,
so the reduction excludes it instead. `ttrl.algos.ppo.loss` takes the reducer
as an argument for exactly this.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ttrl.algos.ppo.loss import masked_mean
from ttrl.engine.autograd import Tensor
from ttrl.engine.carry import ActingCarry

__all__ = ["RecurrentRuntime"]


class RecurrentRuntime:
    """The carries, the start flags and the sequence forward for one PPO."""

    def __init__(self, net, device, *, n_envs: int, seq_len: int):
        self.device = device
        self.seq_len = int(seq_len)
        self.hidden_sizes = dict(net.hidden_sizes)
        self.n_envs = int(n_envs)
        self._carries: dict[tuple[int, bool], dict] = {}
        # The previous training step's `terminated or truncated`, which is the
        # next step's "this state begins an episode".
        self._boundary = np.zeros(self.n_envs, dtype=np.float32)

    # -- acting -------------------------------------------------------------

    def carries(self, lanes: int, *, deterministic: bool) -> dict:
        """The carries for this width and purpose, built on first sight."""
        key = (int(lanes), bool(deterministic))
        carries = self._carries.get(key)
        if carries is None:
            carries = self._carries[key] = {
                name: ActingCarry(size, self.device, lanes=int(lanes),
                                  name=f"{name}@{lanes}")
                for name, size in self.hidden_sizes.items()
            }
        return carries

    def start_flags(self, lanes: int, episode_starts, *,
                    deterministic: bool) -> np.ndarray:
        """The flags to mask the state entering this step with."""
        if episode_starts is not None:
            flags = np.asarray(episode_starts, dtype=np.float32).reshape(-1)
            if flags.shape != (int(lanes),):
                raise ValueError(
                    f"PPO.act: episode_starts must hold one flag per lane "
                    f"({lanes},), got shape {flags.shape}"
                )
            return flags
        if deterministic:
            raise ValueError(
                "PPO.act: this policy is recurrent, so a deterministic acting "
                "call must say which lanes are starting a fresh episode. PPO "
                "sees no transitions during evaluation, so it cannot derive "
                "them: pass episode_starts, the previous step's `terminated or "
                "truncated` (ttrl.train.evaluate does). Without it the hidden "
                "state would carry across every evaluation episode boundary "
                "and the score would be of a policy nobody runs."
            )
        if int(lanes) != self.n_envs:
            raise ValueError(
                f"PPO.act: a training acting call must cover all {self.n_envs} "
                f"lanes, got {lanes}; the episode-start flags PPO tracks are "
                f"lane-major over its own vector env"
            )
        return self._boundary

    def note_boundary(self, done) -> None:
        """Record this training step's episode boundaries for the next step."""
        self._boundary = np.asarray(done, dtype=np.float32).reshape(-1).copy()

    def step(self, acting_net, observation, *, lanes: int, episode_starts,
             deterministic: bool, snapshot: bool):
        """One acting timestep. Returns `(policy_out, value, hidden)`.

        `hidden` is the state entering this step, per memory, when `snapshot`
        asks for it, which the rollout storage wants once per sequence. It is
        read off the carry BEFORE the commit and is the unmasked state, which
        is what the unroll expects to be handed as `h0`.
        """
        carries = self.carries(lanes, deterministic=deterministic)
        flags = self.start_flags(lanes, episode_starts,
                                 deterministic=deterministic)
        for carry in carries.values():
            carry.load_start(flags)
        hidden = ({name: carry.numpy() for name, carry in carries.items()}
                  if snapshot else None)
        policy_out, value, h_next = acting_net.step(
            observation,
            {name: carry.prev for name, carry in carries.items()},
            {name: carry.start for name, carry in carries.items()},
        )
        for name, carry in carries.items():
            carry.bind(h_next[name])
        return policy_out, value, hidden

    def commit(self, lanes: int, *, deterministic: bool) -> None:
        """Advance the carries, after every host read of the step's outputs."""
        for carry in self.carries(lanes, deterministic=deterministic).values():
            carry.commit()

    def bootstrap(self, acting_net, observation, *, lanes: int):
        """The critic's value for the state after the rollout's last step.

        The step runs without binding or committing, so the carry does not
        advance: this observation is read for a bootstrap and never acted on,
        and the next acting step must still start from the state the last real
        step produced.

        NO RESET HERE, even when the last step ended an episode. This
        observation is that step's `next_obs`, and under the NEXT_STEP
        autoreset contract a step that reports terminated or truncated returns
        its own episode's FINAL observation; the post-reset one arrives on the
        following step, which belongs to the next rollout and gets its reset
        from the carried flag. So the state entering this observation is the
        ending episode's own state. Masking it here would value the final
        observation of a truncated episode from a blank memory, which is the
        one bootstrap a truncation actually depends on.
        """
        carries = self.carries(lanes, deterministic=False)
        for carry in carries.values():
            carry.load_start(np.zeros(int(lanes), dtype=np.float32))
        _policy_out, value, _h_next = acting_net.step(
            observation,
            {name: carry.prev for name, carry in carries.items()},
            {name: carry.start for name, carry in carries.items()},
        )
        return value

    def zero(self) -> None:
        """Clear every carry and every pending start flag.

        For a checkpoint restore: PPO already discards the half-collected
        rollout, on the ground that it belongs to a policy that no longer
        exists, and the hidden state it was collected with is part of that
        same discarded context.
        """
        for carries in self._carries.values():
            for carry in carries.values():
                carry.zero()
        self._boundary = np.zeros(self.n_envs, dtype=np.float32)

    # -- learning -----------------------------------------------------------

    def forward(self, acting_net, batch, device):
        """The sequence forward for one minibatch, its reducer and the batch.

        The batch comes back because an invalid slot's learning signal is
        zeroed on the way through. Masking by multiplication is correct for
        finite numbers and not for anything else: `0.0 * inf` is NaN, so one
        non-finite value in a slot the loss excludes would still take the
        whole reduction with it, through the importance ratio in particular,
        which is an exponential of a stored log-prob. Zeroing the three fields
        the loss reads from an excluded slot bounds that ratio and leaves
        every included slot untouched.

        The per-minibatch uploads here are fresh tensors rather than writes
        into bound buffers, which is what PPO's update already does for its
        advantages and returns: this update is not captured as a trace, so
        there is no recorded address for them to stay at. The acting step is
        the captured one and its state lives in `ActingCarry`.
        """
        if batch.seq_len != self.seq_len:
            raise ValueError(
                f"PPO: the minibatch holds sequences of {batch.seq_len} steps "
                f"and this policy unrolls {self.seq_len}"
            )
        live = float(batch.valid.sum())
        if live <= 0.0:
            raise ValueError(
                "PPO: a sequence minibatch holds no valid transitions, so every "
                "loss term would reduce over nothing. Every slot in it is "
                "autoreset padding, which means the rollout is almost entirely "
                "padding; raise n_steps or check the env's autoreset contract."
            )
        rows = len(batch)
        keep = batch.valid.reshape(rows).astype(np.float32)
        batch = dataclasses.replace(
            batch,
            advantages=(batch.advantages * keep).astype(np.float32),
            returns=(batch.returns * keep).astype(np.float32),
            old_logp=(batch.old_logp * keep).astype(np.float32),
        )
        starts = Tensor.from_numpy(
            np.ascontiguousarray(batch.starts, dtype=np.float32), device)
        h0 = {name: Tensor.from_numpy(
                  np.ascontiguousarray(array, dtype=np.float32), device)
              for name, array in batch.h0.items()}
        valid = Tensor.from_numpy(
            batch.valid.reshape(rows, 1).astype(np.float32), device)
        policy_out, value = acting_net(batch.obs, h0, starts,
                                       seq_len=batch.seq_len)
        return policy_out, value, masked_mean(valid), batch

    def close(self) -> None:
        for carries in self._carries.values():
            for carry in carries.values():
                carry.close()
        self._carries.clear()
