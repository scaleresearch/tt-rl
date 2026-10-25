"""What the two off-policy algorithms share: writing a vector step into replay,
the fixed device buffers one sampled batch is loaded into, and the two ends of
prioritized replay that are not either algorithm's arithmetic.

Both DQN and SAC see the same transitions and keep the same storage, so the
lane bookkeeping, the autoreset phase and the sampled-batch buffers live here
once. The replay-ratio arithmetic lives in `ttrl.algos.clock`, which stays
free of ttnn so a host test can check it.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.base import AutoresetTracker, Transition
from ttrl.buffers.sampling import beta_at
from ttrl.engine.autograd import Tensor
from ttrl.engine.tensor import restore_in_place
from ttrl.nets.obs import ObsViews
from ttrl.spaces import ActionSpec, ObsGroup, require_group

__all__ = ["BatchBuffers", "ReplayWriter", "sample_for_update",
           "write_back_priorities"]


class ReplayWriter:
    """Writes the real lanes of a `Transition` into a `ReplayBuffer`.

    Lane by lane rather than through `add_batch`, because one lane's autoreset
    padding step coincides with the other lanes' real steps and `add_batch`
    writes every lane or none.
    """

    def __init__(self, buffer, n_envs: int):
        self.buffer = buffer
        self.n_envs = int(n_envs)
        self._autoreset = AutoresetTracker(self.n_envs)

    def observe(self, transition: Transition) -> None:
        valid = self._autoreset.valid(transition)
        done = transition.done
        for lane in np.flatnonzero(valid):
            ended = bool(done[lane])
            self.buffer.add(
                transition.obs.select(lane), transition.action[lane],
                float(transition.reward[lane]), bool(transition.terminated[lane]),
                bool(transition.truncated[lane]),
                # Under NEXT_STEP the observation a terminating step returns IS
                # that episode's final one, so next_obs carries it.
                final_obs=(transition.next_obs.select(lane) if ended else None),
                lane=int(lane),
            )
            # What follows this write, for the one column that has no
            # successor in the ring yet. `ReplayBuffer.close_open_episodes`
            # is what it is for.
            self.buffer.note_open_successor(
                int(lane), None if ended else transition.next_obs.select(lane))

    def state_dict(self) -> dict:
        return self._autoreset.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._autoreset.load_state_dict(state)


class BatchBuffers:
    """Fixed-address device buffers for one sampled replay batch.

    Address stability: each buffer is allocated once here and thereafter only
    written into, never rebound, so a captured trace keeps replaying against
    the address it recorded while still seeing each new batch.

    `obs` and `next_obs` are `ObsPair`s rather than entries of `tensors`: each
    carries the actor's half of the batch and the critic's as a mapping from
    leaf name to device tensor, and the network reading one takes the half it
    was built over. They are the same mapping when nothing is critic-only.
    """

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, device, *,
                 batch: int, extra: dict[str, tuple[int, ...]] | None = None):
        self.obs_spec = require_group(obs_spec, "BatchBuffers")
        self.batch = int(batch)
        self._views = {name: ObsViews(obs_spec, device, batch=self.batch)
                       for name in ("obs", "next_obs")}
        self.obs = self._views["obs"].pair
        self.next_obs = self._views["next_obs"].pair
        shapes: dict[str, tuple[int, ...]] = {
            "rewards": (self.batch, 1),
            "dones": (self.batch, 1),
            "discounts": (self.batch, 1),
            "actions": (self.batch, action_spec.width),
        }
        shapes.update(extra or {})
        self.tensors = {
            name: Tensor.from_numpy(np.zeros(shape, np.float32), device)
            for name, shape in shapes.items()
        }

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def load(self, name: str, array) -> None:
        restore_in_place(self.tensors[name].data, np.asarray(array, np.float32))

    def load_obs(self, name: str, obs) -> None:
        """Write a host observation batch into both halves of `name`."""
        self._views[name].load(obs)

    def close(self) -> None:
        """Release the observation staging buffers. Idempotent."""
        for views in self._views.values():
            views.close()

    def load_batch(self, batch) -> None:
        """Load the spec-shaped fields every off-policy algorithm reads."""
        self.load_obs("obs", batch.obs)
        self.load_obs("next_obs", batch.next_obs)
        self.load("rewards", batch.rewards.reshape(-1, 1))
        self.load("dones", batch.dones.reshape(-1, 1))
        self.load("discounts", batch.discounts.reshape(-1, 1))


def sample_for_update(buffer, cfg, update_count: int):
    """One batch for a gradient step, under whichever sampler the run asked for.

    Shared by DQN and SAC because the arithmetic is the same in both: the
    importance-sampling exponent is a function of the gradient-step count,
    which the agent already owns and already checkpoints, so neither the
    buffer nor this holds a second clock for it.
    """
    if cfg.priority is None:
        return buffer.sample(cfg.batch_size)
    return buffer.sample(cfg.batch_size, beta=beta_at(cfg.priority, update_count))


def write_back_priorities(buffer, batch, td_abs) -> None:
    """Record a batch's fresh TD errors against the columns they came from.

    THE HOST HALF OF PER. The device half ends when the metrics of the
    captured step are read back; this is a numpy write into the priority
    table and belongs nowhere near a captured region. It runs before the next
    `observe`, so no column has been overwritten between the draw and this.
    """
    buffer.update_priorities(batch.lanes, batch.cols,
                             np.asarray(td_abs).reshape(-1))
