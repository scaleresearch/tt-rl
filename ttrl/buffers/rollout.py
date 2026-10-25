"""On-policy rollout storage with GAE-lambda.

Laid out `[T, n_envs, ...]` so the advantage recursion over time never reaches
across the lane axis into another environment's trajectory: indexing is always
`buf[t]` for a whole timestep and the recursion walks `t` from `T-1` down to
`0` with a per-lane running `last_gae` vector.

Shapes come from the `ObsGroup` and the `ActionSpec`, so a continuous action is
stored as the float vector it is rather than coerced to an integer id, and an
observation is stored as one array per leaf, each of its own dtype. A
single-array observation is one leaf.
"""

from __future__ import annotations

import numpy as np

from ttrl.buffers.seqbatch import SequenceBatcher
from ttrl.buffers.sequences import chunk_count, store_hidden
from ttrl.buffers.types import (
    RolloutMinibatch,
    action_dtype,
    action_element_shape,
    obs_element_shape,
)
from ttrl.spaces import (ActionSpec, ObsGroup, check_legal, group_mask_rows,
                         require_group)

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """Fixed `[n_steps, n_envs]` on-policy buffer."""

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, *, n_envs: int,
                 n_steps: int, seed: int | None = None,
                 seq_len: int | None = None, hidden_sizes: dict | None = None):
        if n_envs < 1 or n_steps < 1:
            raise ValueError(
                f"RolloutBuffer: n_envs and n_steps must be >= 1, got "
                f"n_envs={n_envs}, n_steps={n_steps}"
            )
        self.obs_spec = require_group(obs_spec, "RolloutBuffer")
        self.action_spec = action_spec
        self.n_envs = int(n_envs)
        self.n_steps = int(n_steps)
        shape = (self.n_steps, self.n_envs)

        #: One array per leaf, every leaf written at the same timestep and lane.
        self.obs = {
            name: np.zeros((*shape, *obs_element_shape(spec)), dtype=spec.numpy_dtype)
            for name, spec in self.obs_spec.specs()
        }
        self.actions = np.zeros(
            (*shape, *action_element_shape(action_spec)), dtype=action_dtype(action_spec)
        )
        self.old_logp = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        # `terminated` only. A time-limit truncation still bootstraps, and under
        # gymnasium's NEXT_STEP autoreset the value it needs is `values[t+1]`,
        # because the truncating step returns the episode's true final
        # observation and the following padding step carries the post-reset one.
        self.dones = np.zeros(shape, dtype=np.float32)
        # 1.0 for a real transition, 0.0 for an autoreset padding step whose
        # action the env ignored and whose reward and done are fabricated zeros.
        # Those slots never reach an update and never propagate an advantage,
        # but they still hold the honest V(s_final) the preceding step reads.
        self.valid = np.ones(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        # 1.0 where a step BEGINS an episode, which is the previous step's
        # `terminated or truncated` and not this step's. See `add`.
        self.starts = np.zeros(shape, dtype=np.float32)

        if (seq_len is None) != (hidden_sizes is None):
            raise ValueError(
                f"RolloutBuffer: seq_len and hidden_sizes are both the recurrent "
                f"case and must be given together, got seq_len={seq_len} and "
                f"hidden_sizes={hidden_sizes}"
            )
        self.seq_len = None if seq_len is None else int(seq_len)
        self.hidden_sizes = dict(hidden_sizes or {})
        self.n_chunks = (0 if self.seq_len is None
                         else chunk_count(self.n_steps, self.seq_len))
        if self.recurrent and not self.hidden_sizes:
            raise ValueError(
                "RolloutBuffer: a recurrent buffer needs at least one memory in "
                "hidden_sizes; the sequences would otherwise carry no state"
            )
        #: memory name -> the state entering each sequence's first step, one
        #: row per lane. Host memory, so a stored hidden state raises no
        #: address-stability question at all: the only device buffer in the
        #: carry is the one `ttrl.engine.carry.ActingCarry` owns.
        self.hidden = {
            name: np.zeros((self.n_chunks, self.n_envs, int(size)), dtype=np.float32)
            for name, size in self.hidden_sizes.items()
        }

        self._t = 0
        # Carried ACROSS rollouts, so `reset` leaves it alone: the first step
        # of a rollout begins an episode exactly when the last step of the
        # previous one ended one.
        self._carried_boundary = np.zeros(self.n_envs, dtype=np.float32)
        self._gae_computed = False
        self._rng = np.random.default_rng(seed)

    @property
    def full(self) -> bool:
        return self._t >= self.n_steps

    @property
    def recurrent(self) -> bool:
        return self.seq_len is not None

    @property
    def at_sequence_start(self) -> bool:
        """Whether the next `add` opens a new sequence, and so wants a hidden
        state. The acting loop asks this before it steps, because the state a
        sequence starts from is the one ENTERING its first step."""
        return self.recurrent and self._t % self.seq_len == 0

    def reset(self) -> None:
        self._t = 0
        self._gae_computed = False
        self.valid[:] = 1.0

    def add(self, obs, actions, old_logp, values, rewards, dones, valid=None, *,
            boundary=None, hidden=None) -> None:
        """Append one timestep. Every argument's leading axis is the lane.

        `dones` IS `terminated` ALONE AND `boundary` IS NOT. A time-limit
        truncation must still bootstrap, which is why `dones` carries
        terminations only, and the hidden state must still reset, because the
        step after a truncation belongs to a different episode. Handing
        `dones` where `boundary` belongs therefore leaks hidden state across
        every truncation in the run and nothing anywhere reports it. A
        recurrent buffer refuses a `boundary` of None for that reason, and
        `tests/buffers/test_rollout_starts.py` fails when the two are
        swapped, and `tests/hardware/test_recurrent_update.py` fails when PPO
        passes the wrong one.

        `hidden` is `{memory name: (n_envs, size)}`, the state entering this
        step, required exactly at a sequence's first step and refused
        elsewhere.
        """
        if self.full:
            raise RuntimeError(
                f"RolloutBuffer.add: buffer already holds its {self.n_steps} steps; "
                "call reset() before collecting another rollout"
            )
        t = self._t
        observation = self.obs_spec.observation(obs)
        if self.obs_spec.action_mask:
            check_legal(observation, actions, self.obs_spec, "RolloutBuffer.add")
        for name, array in self.obs.items():
            array[t] = observation[name]
        self.actions[t] = actions
        self.old_logp[t] = old_logp
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.valid[t] = 1.0 if valid is None else valid
        self._record_boundary(t, boundary)
        self._record_hidden(t, hidden)
        self._t += 1

    def _record_boundary(self, t: int, boundary) -> None:
        if boundary is None:
            if self.recurrent:
                raise ValueError(
                    "RolloutBuffer.add: a recurrent buffer needs this step's "
                    "episode boundary (terminated OR truncated) to know where "
                    "the hidden state resets. `dones` holds terminations alone, "
                    "so it is not that flag: passing it would carry state "
                    "across every time-limit truncation."
                )
            flags = np.zeros(self.n_envs, dtype=np.float32)
        else:
            flags = np.asarray(boundary, dtype=np.float32).reshape(-1)
            if flags.shape != (self.n_envs,):
                raise ValueError(
                    f"RolloutBuffer.add: boundary must be lane-major with "
                    f"{self.n_envs} entries, got shape {flags.shape}"
                )
        self.starts[t] = self._carried_boundary
        # In place: `flags` may be a caller's own row (PPO's boundary seam
        # hands one over per step), and holding a reference to it would let a
        # later write change a boundary already carried.
        self._carried_boundary[:] = flags

    def _record_hidden(self, t: int, hidden) -> None:
        store_hidden(self.hidden, t, self.seq_len, hidden)

    def compute_gae(self, gamma: float, lam: float, last_values) -> None:
        """GAE-lambda, vectorised over lanes, recursing `t: T-1 -> 0`.

        `last_values` is the critic's estimate for the state after the last
        stored step, shape `(n_envs,)`. `advantages` and
        `returns = advantages + values` are written back into this buffer's own
        arrays.
        """
        if not self.full:
            raise RuntimeError(
                f"RolloutBuffer.compute_gae: buffer holds {self._t} of {self.n_steps} "
                "steps; the recursion walks every slot, so unwritten ones would fold "
                "stale or zero transitions into the advantages"
            )
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        next_values = np.asarray(last_values, dtype=np.float32)
        if next_values.shape != (self.n_envs,):
            raise ValueError(
                f"RolloutBuffer.compute_gae: last_values must have shape "
                f"({self.n_envs},), got {next_values.shape}"
            )
        for t in range(self.n_steps - 1, -1, -1):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            last_gae = delta + gamma * lam * not_done * last_gae
            # A padding slot carries no advantage and cuts the chain, so nothing
            # downstream of it leaks into the real transition before it.
            last_gae = last_gae * self.valid[t]
            self.advantages[t] = last_gae
            next_values = self.values[t]
        self.returns[:] = self.advantages + self.values
        self._gae_computed = True

    # -- minibatches --------------------------------------------------------

    def _flat(self, name: str) -> np.ndarray:
        arr = getattr(self, name)
        return arr.reshape(self.n_steps * self.n_envs, *arr.shape[2:])

    def _flat_obs(self, keep: np.ndarray) -> dict[str, np.ndarray]:
        """The kept transitions' observations, flattened, leaf by leaf."""
        return {
            name: array.reshape(self.n_steps * self.n_envs, *array.shape[2:])[keep]
            for name, array in self.obs.items()
        }

    def minibatches(self, minibatch_size: int, epochs: int = 4, normalize_adv: bool = True):
        """Yield shuffled flattened minibatches, `epochs` passes over the data.

        Only valid slots are drawn, so autoreset padding never reaches an
        update or enters the advantage statistics. The minibatch size is fixed
        because every minibatch is uploaded into a stable-shape device arena,
        so a trailing partial batch is dropped rather than yielded ragged.
        """
        if not self._gae_computed:
            raise RuntimeError("RolloutBuffer.minibatches: call compute_gae() first")
        keep = np.flatnonzero(self._flat("valid") > 0.0)
        total = int(keep.size)
        n_full = total // minibatch_size
        if n_full == 0:
            raise ValueError(
                f"RolloutBuffer.minibatches: {total} valid transitions but "
                f"minibatch_size={minibatch_size}, which would yield zero minibatches "
                f"(n_steps={self.n_steps}, n_envs={self.n_envs}). Lower minibatch_size "
                f"or collect a longer rollout."
            )
        fields = {
            name: self._flat(name)[keep]
            for name in ("actions", "old_logp", "values", "returns")
        }
        obs = self._flat_obs(keep)
        mask = (group_mask_rows(obs, self.obs_spec)
                if self.obs_spec.action_mask else None)
        advantages = self._flat("advantages")[keep]
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            perm = self._rng.permutation(total)
            for start in range(0, n_full * minibatch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                yield RolloutMinibatch(
                    obs=self.obs_spec.observation(
                        {name: array[idx] for name, array in obs.items()}),
                    actions=fields["actions"][idx],
                    old_logp=fields["old_logp"][idx], values=fields["values"][idx],
                    advantages=advantages[idx], returns=fields["returns"][idx],
                    mask=None if mask is None else mask[idx],
                )

    def sequence_minibatches(self, minibatch_size: int, epochs: int = 4,
                             normalize_adv: bool = True):
        """Yield shuffled minibatches of whole sequences, for a recurrent
        update. Assembly lives in `ttrl.buffers.seqbatch`."""

        return SequenceBatcher(self).minibatches(
            minibatch_size, epochs=epochs, normalize_adv=normalize_adv)
