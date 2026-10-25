"""The PPO agent: clipped surrogate, GAE advantages, entropy bonus, value
clipping, gradient-norm clipping and linear learning-rate annealing.

One rollout is one `update()`: the buffer fills, every epoch's minibatches run,
and the buffer resets. The cadence is `should_update`'s answer and the trainer
never learns that this algorithm is on-policy.

`act` and `observe` are a pair. `act` records the log-prob and the value it
computed, because recomputing them in `observe` would be a second forward pass
over the same observations, and the importance ratio must be measured against
the log-prob the acting policy actually drew with. `observe` therefore refuses
a transition that no `act` preceded.

The update body lives in `ttrl.algos.ppo.learn`: this file holds the acting
pair, the rollout and the boundary seam those share, and that one holds the
shape of one gradient step.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.base import AutoresetTracker, Transition
from ttrl.algos.mask import mask_width, obs_mask
from ttrl.algos.ppo import learn
from ttrl.algos.ppo.boundaries import read_boundaries, require_boundaries
from ttrl.algos.ppo.build import build_net
from ttrl.algos.ppo.config import PPOConfig
from ttrl.algos.ppo.policy import build_policy
from ttrl.algos.ppo.recurrent import RecurrentRuntime
from ttrl.algos.ppo.state import RUN_KEYS, WEIGHT_KEYS, load_into, state_of
from ttrl.algos.scaling import check_network_batch
from ttrl.buffers import RolloutBuffer
from ttrl.engine.autograd import no_grad
from ttrl.engine.optim import Adam
from ttrl.nets.acting import ActingNet, check_encoder_layout
from ttrl.spaces import ActionSpec, ObsGroup, require_group

__all__ = ["PPO"]


class PPO:
    """Clipped PPO over a shared-trunk actor-critic. Every action kind.

    `PPOConfig.recurrent` puts a GRU between trunk and heads and makes the
    update run over sequences; `ttrl.algos.ppo.recurrent` holds everything
    that then differs.
    """

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, device, *,
                 cfg: PPOConfig = PPOConfig(), n_envs: int = 1, seed: int | None = None,
                 profile=None, boundaries=None):
        check_network_batch(cfg, n_envs, "ppo")
        # Before any network is built: a run that would anneal against no
        # horizon is refused here rather than at the first update, by which
        # time a chip is open and a rollout has been collected.
        cfg.require_horizon()
        if cfg.minibatch_size > cfg.n_steps * n_envs:
            raise ValueError(
                f"PPO: minibatch_size={cfg.minibatch_size} exceeds the "
                f"{cfg.n_steps * n_envs} transitions one rollout collects "
                f"(n_steps={cfg.n_steps} x n_envs={n_envs}), so no minibatch "
                f"could ever be drawn"
            )
        self.obs_spec = require_group(obs_spec, "ppo")
        self.action_spec = action_spec
        self.device = device
        self.profile = profile
        self.cfg = cfg
        self.n_envs = int(n_envs)
        # One switch: the sampling in `act` and the log-prob and entropy in
        # the update body both read it.
        self.mask_n = mask_width(obs_spec, action_spec, "ppo")

        if seed is not None:
            np.random.seed(seed)
        memory = cfg.recurrent
        self.net = build_net(obs_spec, action_spec, device, cfg)
        check_encoder_layout(self.net)
        self.policy = build_policy(action_spec, device)
        self.optimizer = Adam(list(self.net.parameters()), lr=cfg.lr)

        self.buffer = RolloutBuffer(
            obs_spec, action_spec, n_envs=self.n_envs, n_steps=cfg.n_steps,
            seed=seed,
            seq_len=None if memory is None else memory.seq_len,
            hidden_sizes=None if memory is None else self.net.hidden_sizes,
        )
        # Where this policy's memory resets, declared rather than derived:
        # `ttrl.algos.ppo.boundaries` is the seam and says what it refuses.
        self.boundaries = require_boundaries(
            boundaries, recurrent=memory is not None, who="PPO")
        self._autoreset = AutoresetTracker(self.n_envs)
        # Acting and the bootstrap forward run at the lane count; only the
        # minibatch forward wants the full minibatch_size width.
        self._obs = ActingNet(self.net, obs_spec, device,
                              width=cfg.minibatch_size, lanes=self.n_envs)
        self._recurrent = None
        if memory is not None:
            self._recurrent = RecurrentRuntime(
                self.net, device, n_envs=self.n_envs, seq_len=memory.seq_len)
        self._rng = np.random.default_rng(seed)
        self._eval_rng = np.random.default_rng(seed).spawn(1)[0]
        self._pending: dict | None = None
        self._last_next_obs = None
        self.frame_count = 0
        self.update_count = 0

    # -- acting ------------------------------------------------------------

    def act(self, obs, *, deterministic: bool = False,
            episode_starts=None) -> np.ndarray:
        """One action per lane. `episode_starts` is lane-major and 1.0 where
        that lane's PREVIOUS step ended an episode, which is where a recurrent
        policy's hidden state resets; a memoryless one has nothing to reset.
        `ttrl.algos.ppo.recurrent` says where the flags come from."""
        observation = self.obs_spec.observation(obs)
        batch = int(observation.lanes)
        mask = obs_mask(observation, self.obs_spec, self.mask_n)
        hidden = None
        with no_grad():
            if self._recurrent is None:
                policy_out, value = self._obs(observation)
            else:
                policy_out, value, hidden = self._recurrent.step(
                    self._obs, observation, lanes=batch,
                    episode_starts=episode_starts, deterministic=deterministic,
                    snapshot=not deterministic and self.buffer.at_sequence_start,
                )
        rng = self._eval_rng if deterministic else self._rng
        env_action, stored, logp = self.policy.sample(
            policy_out, rng, batch, deterministic, mask=mask)
        if self._recurrent is not None:
            # After every host read of this step's outputs, and outside any
            # captured region: the carry advances by a device-to-device copy.
            self._recurrent.commit(batch, deterministic=deterministic)
        if not deterministic:
            self._pending = {
                "action": stored,
                "logp": logp,
                "value": value.numpy().reshape(-1)[:batch].astype(np.float32),
                "hidden": hidden,
            }
        return env_action

    # -- observing ---------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        """Append one vector step to the rollout, padding steps included.

        A padding step is stored with valid=0: it never reaches an update and
        never propagates an advantage, but the value it holds is V of the
        preceding episode's final observation, which is exactly the bootstrap
        a truncated episode's last real step needs.
        """
        if self._pending is None:
            raise RuntimeError(
                "PPO.observe: no act() preceded this transition. PPO records the "
                "log-prob and the value from the forward pass that chose the "
                "action, so every observe must follow the act that produced it; "
                "a deterministic act (evaluation) records nothing and must not "
                "be fed back."
            )
        valid = self._autoreset.valid(transition)
        pending, self._pending = self._pending, None
        if pending["logp"].shape[0] != transition.n_envs:
            raise ValueError(
                f"PPO.observe: the last act() produced "
                f"{pending['logp'].shape[0]} actions but this transition has "
                f"{transition.n_envs} lanes"
            )
        recurrent = self._recurrent is not None
        # ONE read of the seam, two consumers: the rollout's `starts`, which
        # the unroll masks with, and the acting carry's reset for the next
        # step. Acting and learning resetting at different steps trains a
        # policy nobody evaluates and nothing reports it. `dones` is
        # `terminated` alone and a boundary is not; RolloutBuffer.add says why.
        flags = (read_boundaries(self.boundaries, transition,
                                 n_envs=self.n_envs) if recurrent else None)
        self.buffer.add(
            obs=transition.obs, actions=pending["action"], old_logp=pending["logp"],
            values=pending["value"], rewards=transition.reward,
            dones=np.asarray(transition.terminated, np.float32),
            valid=valid.astype(np.float32),
            boundary=flags,
            hidden=pending.get("hidden"),
        )
        if recurrent:
            self._recurrent.note_boundary(flags)
        self._last_next_obs = transition.next_obs
        self.frame_count += transition.n_envs

    def should_update(self, frame: int) -> bool:
        """A full rollout, which is what an on-policy update consumes. The
        frame number is irrelevant: the trigger is how much data this policy
        has collected, not where the run has got to."""
        return self.buffer.full

    # -- replica averaging -------------------------------------------------

    def named_parameters(self):
        """The whole actor-critic: on-policy learning has no lagged copy."""
        return self.net.named_parameters()

    def parameters(self):
        return self.net.parameters()

    # -- learning ----------------------------------------------------------

    # `ttrl.algos.ppo.learn` is the update body. These stay as methods because
    # they are what the trainer and the parity tests call.

    def _last_values(self) -> np.ndarray:
        return learn.last_values(self)

    def update(self) -> dict[str, float]:
        return learn.update(self)

    def _minibatch_step(self, batch) -> dict[str, float]:
        return learn.minibatch_step(self, batch)

    # -- checkpointing -----------------------------------------------------

    _WEIGHT_KEYS = WEIGHT_KEYS
    _RUN_KEYS = RUN_KEYS

    def state_dict(self, *, weights_only: bool = False) -> dict:
        return state_of(self, weights_only=weights_only)

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        load_into(self, state, weights_only=weights_only)

    def close(self) -> None:
        """Release the acting buffer's staging and any carry. PPO holds no
        captured trace."""
        self._obs.close()
        if self._recurrent is not None:
            self._recurrent.close()
