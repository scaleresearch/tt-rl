"""The replay buffer. One class, always lane-dimensioned.

`n_envs` is a real axis of the storage, with `n_envs == 1` the degenerate case
rather than a separate class. Lanes are the general case because every piece of
ring arithmetic (the forward n-step walk, the wrap seam, the neighbour read
below) is only correct when neighbouring columns are consecutive steps of one
stream. Interleaving several vector-env streams into one ring would fabricate
returns and successor states out of steps that never followed one another, and
nothing in the stored data could detect it afterwards, so the lane axis
prevents it structurally. See `ttrl.buffers.indexing` for the invariant.

Element shapes and dtypes come from the `ObsGroup` and the `ActionSpec`.
Nothing here knows a frame size, a stack depth or a pixel dtype: frame
stacking is an env-side wrapper concern, and this buffer stores whatever the
spec declares one observation to be.

ONE ARRAY PER LEAF, and a single-array observation is one leaf. Every leaf is
stored in its own array of its own dtype, because a camera frame and a state
vector share neither, and every leaf is written and read at the same (lane,
column), so an observation is never assembled from two different steps.

Column `i` of a lane holds the state its action was taken from, so the
successor of transition `i` is the observation in column `i + 1` of the same
lane. The exception is an episode boundary, whose successors come from
`final_obs` and live in `ttrl.buffers.boundaries`. The cost is one array per
leaf plus a handful of episode-final observations rather than two full sets:
at 1M transitions of 4x84x84 uint8 that is 28 GB instead of 56 GB.

The read that makes this work spans `n_step + 1` columns, so the sampleable run
is one column shorter than the window alone would need. Getting that bound
wrong reads an observation the write cursor has already overwritten, which is
wrong training data and not an error, so it lives in `indexing.sampleable`
with the derivation written out and is tested at the seam directly.
"""

from __future__ import annotations

import numpy as np

from ttrl.buffers.boundaries import BoundaryTable
from ttrl.buffers.indexing import sampleable
from ttrl.buffers.openepisodes import close_open_episodes, note_open_successor
from ttrl.buffers.replaybatch import assemble
from ttrl.buffers.sampling import PriorityConfig
from ttrl.buffers.samplers import PriorityFeedback, build_sampler
from ttrl.buffers.window import SampleWindow
from ttrl.buffers.writes import check_add, check_add_batch
from ttrl.buffers.types import (
    ReplayBatch,
    action_dtype,
    action_element_shape,
    obs_element_shape,
)
from ttrl.spaces import ActionSpec, ObsGroup, check_legal, require_group

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Uniform off-policy replay with n-step returns, over `n_envs` lanes."""

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, *, capacity: int,
                 n_envs: int = 1, n_step: int = 1, gamma: float = 0.99,
                 seed: int | None = None, priority: PriorityConfig | None = None,
                 sampler=None):
        if n_envs < 1:
            raise ValueError(f"ReplayBuffer: n_envs must be >= 1, got {n_envs}")
        if n_step < 1:
            raise ValueError(f"ReplayBuffer: n_step must be >= 1, got {n_step}")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"ReplayBuffer: gamma must be in (0, 1], got {gamma}")
        per_lane = -(-int(capacity) // int(n_envs))  # ceil: capacity is split, not multiplied
        if per_lane <= n_step + 1:
            raise ValueError(
                f"ReplayBuffer: capacity {capacity} split across {n_envs} lanes gives "
                f"{per_lane} columns per lane, but one transition reads n_step + 1 = "
                f"{n_step + 1} columns of its own lane (the n-step window plus the "
                f"neighbouring observation), so a lane that short has no sampleable "
                f"transition. Raise capacity or lower n_envs."
            )
        self.obs_spec = require_group(obs_spec, "ReplayBuffer")
        self.action_spec = action_spec
        self.n_envs = int(n_envs)
        self.per_lane_capacity = per_lane
        self.capacity = per_lane * self.n_envs
        self.n_step = int(n_step)
        self.gamma = float(gamma)

        lanes, cols = self.n_envs, per_lane
        self._obs_shapes = {name: obs_element_shape(spec)
                            for name, spec in self.obs_spec.specs()}
        #: One array per leaf, each at the same (lane, column) as the others.
        self.obs = {
            name: np.zeros((lanes, cols, *shape), dtype=self.obs_spec[name].numpy_dtype)
            for name, shape in self._obs_shapes.items()
        }
        self.actions = np.zeros(
            (lanes, cols, *action_element_shape(action_spec)), dtype=action_dtype(action_spec)
        )
        self.rewards = np.zeros((lanes, cols), dtype=np.float32)
        # `dones` is gymnasium's `terminated`, the narrow MDP terminal where the
        # bootstrap is zeroed. `ends` is terminated OR truncated and is what the
        # n-step walk stops at: a time-limit cutoff ends the episode without
        # making the state terminal.
        self.dones = np.zeros((lanes, cols), dtype=bool)
        self.ends = np.zeros((lanes, cols), dtype=bool)
        #: Ending columns' final observations. Evicted when its column is
        #: overwritten, so it holds exactly the boundaries still reachable
        #: from the ring. See `ttrl.buffers.boundaries`.
        self._final_obs = BoundaryTable(self.obs_spec)
        #: lane -> the observation that follows its newest column, for the
        #: lanes whose newest column does not end an episode. Not in the ring
        #: until that lane's next write puts it there, so a snapshot taken
        #: mid-episode needs it to close the seam. See `close_open_episodes`.
        self._open: dict[int, dict[str, np.ndarray]] = {}

        self._pos = np.zeros(lanes, dtype=np.int64)
        self._full = np.zeros(lanes, dtype=bool)
        self._rng = np.random.default_rng(seed)
        #: Which transitions a batch is drawn from, and with what weight: a
        #: registered name, an instance, or None for the declared default of
        #: uniform, or prioritized when a run gave priority settings. See
        #: `ttrl.buffers.samplers`.
        self._sampler = build_sampler(sampler, lanes=lanes, columns=cols,
                                      config=priority)

    # -- size ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Transitions stored, across every lane."""
        return int(np.where(self._full, self.per_lane_capacity, self._pos).sum())

    @property
    def num_sampleable(self) -> int:
        """Transitions `sample` can draw right now.

        A warm-up gate must check this and not `size`: with many lanes, `size`
        can clear a batch size while no lane has yet written a full n-step
        window plus its successor observation, leaving nothing to draw.
        """
        _, count = sampleable(self._pos, self._full, self.per_lane_capacity, self.n_step)
        return int(count.sum())

    @property
    def num_boundaries(self) -> int:
        """Live entries in the sparse episode-boundary table."""
        return len(self._final_obs)

    # -- writing ------------------------------------------------------------

    def add(self, obs, action, reward, terminated: bool, truncated: bool = False, *,
            final_obs=None, lane: int = 0) -> None:
        """Record one transition into one lane.

        `obs` is the state the action was taken from. The successor state is
        the next call's `obs`, except when this step ends the episode, where
        `final_obs` is that episode's last observation and is required: the
        following call's `obs` belongs to a different episode, so without it
        this transition's successor is unrecoverable rather than merely
        unknown.
        """
        ended = check_add(lane, self.n_envs, terminated, truncated, final_obs)
        col = int(self._pos[lane])
        self._write_slot(lane, col, obs, action, reward, terminated, ended, final_obs)
        self._advance(lane)

    def add_batch(self, obs, actions, rewards, terminated, truncated=None,
                  final_obs=None) -> None:
        """Record one vector-env step: the leading axis of every argument is the lane.

        `final_obs` is the env's `info[FINAL_OBS_KEY]`, a sequence of length
        `n_envs` holding each ended lane's last observation and None for every
        lane that did not end, so it is passed straight through from `step`.
        """
        obs = self.obs_spec.observation(obs)
        term, ended = check_add_batch(obs, self.n_envs, terminated, truncated,
                                      final_obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        for lane in range(self.n_envs):
            final = None if final_obs is None else final_obs[lane]
            self._write_slot(lane, int(self._pos[lane]), obs.select(lane), actions[lane],
                             rewards[lane], term[lane], ended[lane],
                             final if ended[lane] else None)
            self._advance(lane)

    def _write_slot(self, lane: int, col: int, obs, action, reward, terminated,
                    ended, final_obs) -> None:
        # Evict first: the entry under this key belongs to the lap being
        # overwritten, and a stale boundary observation left here would be
        # served as some live transition's successor state.
        self._final_obs.evict(lane, col)
        observation = self.obs_spec.observation(obs)
        if self.obs_spec.action_mask:
            check_legal(observation, action, self.obs_spec, "ReplayBuffer.add")
        for name, array in self.obs.items():
            array[lane, col] = observation[name]
        self.actions[lane, col] = action
        self.rewards[lane, col] = reward
        self.dones[lane, col] = bool(terminated)
        self.ends[lane, col] = bool(ended)
        # A fresh transition's priority, before its own TD error is known.
        self._sampler.on_write(lane, col)
        if ended:
            self._final_obs.store(lane, col, final_obs)

    def note_open_successor(self, lane: int, obs) -> None:
        """Record the observation following `lane`'s newest column, or None.

        See `ttrl.buffers.openepisodes`.
        """
        note_open_successor(self, lane, obs)

    def close_open_episodes(self) -> list[int]:
        """End every lane's unfinished episode at its newest column, as a
        truncation against the successor recorded for it. Returns the lanes
        closed. See `ttrl.buffers.openepisodes`.
        """
        return close_open_episodes(self)

    def _advance(self, lane: int) -> None:
        nxt = (int(self._pos[lane]) + 1) % self.per_lane_capacity
        self._pos[lane] = nxt
        if nxt == 0:
            self._full[lane] = True

    # -- sampling -----------------------------------------------------------

    @property
    def prioritized(self) -> bool:
        # Asked of the sampler's declared capability and not of its class, so
        # a user's own prioritized variant gets its feedback and its
        # checkpoint. See `ttrl.buffers.samplers.PriorityFeedback`.
        return isinstance(self._sampler, PriorityFeedback)

    def priority_state(self):
        """The sampler's priorities, or None for a uniform buffer."""
        return self._sampler.state() if self.prioritized else None

    def load_priority_state(self, state) -> None:
        self._sampler.load(state)

    def update_priorities(self, lanes, cols, priorities) -> None:
        """Record the TD errors a sampled batch produced, by (lane, column).

        A HOST operation, and deliberately outside everything a trace
        captures: it must be called with the batch's own `lanes` and `cols`
        before the next `add`, which is where the agent calls it. A write
        between the two would overwrite a column and then take this update as
        the new transition's priority.
        """
        if not self.prioritized:
            raise ValueError(
                "ReplayBuffer.update_priorities: this buffer samples "
                "uniformly, so a priority would be stored and never read. "
                "Build it with priority=PriorityConfig(...)."
            )
        self._sampler.update(np.asarray(lanes), np.asarray(cols), priorities)

    def sample(self, batch_size: int, *, beta: float | None = None) -> ReplayBatch:
        """One batch. `beta` is the importance-sampling exponent, required by
        a prioritized buffer and refused by a uniform one."""
        if batch_size < 1:
            raise ValueError(f"ReplayBuffer.sample: batch_size must be >= 1, got {batch_size}")
        start, count = sampleable(self._pos, self._full, self.per_lane_capacity, self.n_step)
        if int(count.sum()) <= 0:
            # Named here rather than in the sampler, which is handed the run
            # and not the buffer: this is the one place that can report how
            # much the buffer holds beside how little of it is sampleable.
            raise ValueError(
                f"ReplayBuffer.sample: no lane holds a complete {self.n_step}-step "
                f"window and its successor observation yet (size={self.size}); gate "
                f"on num_sampleable"
            )
        window = SampleWindow(counts=count, start=start,
                              capacity=self.per_lane_capacity)
        draw = self._sampler.draw(self._rng, window, batch_size=batch_size,
                                  beta=beta)
        lanes, cols, weights = window.locate(draw, batch_size=batch_size,
                                             sampler=self._sampler)
        return assemble(self, lanes, cols, weights)
