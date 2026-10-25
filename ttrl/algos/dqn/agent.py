"""The DQN agent: acting, observing, and the bookkeeping around one update.

Double DQN: n-step returns, Huber loss, a target network and an epsilon
schedule. The arithmetic of an update and the device buffers it reads live in
`ttrl.algos.dqn.losses`.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.base import Transition, require_discrete
from ttrl.buffers.persist import load_replay_state, replay_state
from ttrl.algos.checkpoint import assemble, validate
from ttrl.algos.dqn.config import DQNConfig
from ttrl.algos.dqn.losses import leaf_key, step_inputs, update_loss
from ttrl.algos.mask import epsilon_greedy_masked, mask_width, obs_mask
from ttrl.algos.clock import UpdateClock
from ttrl.algos.offpolicy import (ReplayWriter, sample_for_update,
                                  write_back_priorities)
from ttrl.algos.scaling import check_network_batch
from ttrl.buffers import ReplayBuffer
from ttrl.engine.autograd import no_grad
from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.ops import argmax_dim_detached
from ttrl.engine.optim import Adam
from ttrl.engine.step import TrainStep
from ttrl.nets.build import build_q_net
from ttrl.nets.acting import ActingNet, check_encoder_layout
from ttrl.nets.obs import obs_rows
from ttrl.spaces import ActionSpec, ObsGroup, require_group

__all__ = ["DQN"]


class DQN:
    """Double DQN over a spec-built Q network. Discrete actions only."""

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, device, *,
                 cfg: DQNConfig = DQNConfig(), engine: EngineConfig = DEFAULT,
                 n_envs: int = 1, seed: int | None = None, profile=None):
        require_discrete(action_spec, "dqn")
        check_network_batch(cfg, n_envs, "dqn")
        self.obs_spec = require_group(obs_spec, "dqn")
        self.action_spec = action_spec
        self.device = device
        self.profile = profile
        self.cfg = cfg
        self.n_envs = int(n_envs)
        self.num_actions = int(action_spec.n)
        # One switch for both halves of masking: the selection in `act` and the
        # bootstrap in `_step_fn`. Masking either alone is the trap.
        self.mask_n = mask_width(obs_spec, action_spec, "dqn")

        # Layer init draws from the global numpy RNG, so two runs of one config
        # start from different weights unless it is seeded here.
        if seed is not None:
            np.random.seed(seed)
        self._engine = engine
        build = dict(batch=cfg.batch_size, encoder=cfg.encoder,
                     **cfg.encoder_kwargs)
        self.online = build_q_net(obs_spec, action_spec, device, **build)
        self.target = build_q_net(obs_spec, action_spec, device, **build)
        self.target.copy_from(self.online)
        check_encoder_layout(self.online)
        self.optimizer = Adam(list(self.online.parameters()), lr=cfg.lr)

        self.buffer = ReplayBuffer(
            obs_spec, action_spec, capacity=cfg.buffer_capacity, n_envs=self.n_envs,
            n_step=cfg.n_step, gamma=cfg.gamma, seed=seed,
            priority=cfg.priority, sampler=cfg.sampler,
        )
        self._writer = ReplayWriter(self.buffer, self.n_envs)
        self._clock = UpdateClock(cfg.train_freq, cfg.learning_starts)
        # Acting runs at the lane count, not at batch_size: that is a
        # replay-ratio knob, and padding out to it uploads and convolves
        # rows nobody reads.
        self._acting = ActingNet(self.online, obs_spec, device,
                                 width=cfg.batch_size, lanes=self.n_envs)
        self._step = self._build_step()

        self.frame_count = 0
        self.update_count = 0
        self._rng = np.random.default_rng(seed)
        # A separate stream for evaluation acting, so periodic eval does not
        # perturb the training exploration trajectory as a function of how
        # often eval happens.
        self._eval_rng = np.random.default_rng(seed).spawn(1)[0]

    # -- acting ------------------------------------------------------------

    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.frame_count / cfg.eps_decay_frames)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def act(self, obs, *, deterministic: bool = False,
            episode_starts=None) -> np.ndarray:
        """One action per lane. `episode_starts` is part of the acting
        contract for a policy with memory; this one has none to reset."""
        observation = self.obs_spec.observation(obs)
        batch = int(observation.lanes)
        eps = self.cfg.eval_epsilon if deterministic else self.epsilon()
        rng = self._eval_rng if deterministic else self._rng
        mask = obs_mask(observation, self.obs_spec, self.mask_n)
        with no_grad():
            q = self._acting(observation)
            out = (q if mask is not None else argmax_dim_detached(q, dim=-1)).numpy()
        if mask is not None:
            actions = epsilon_greedy_masked(
                out.reshape(-1, self.num_actions)[:batch], mask, eps, rng)
        else:
            draws = rng.random(batch)
            random_actions = rng.integers(0, self.num_actions, size=batch)
            actions = np.where(draws < eps, random_actions,
                               out.reshape(-1)[:batch]).astype(np.int64)
        if not deterministic:
            self.frame_count += batch
        return actions

    # -- observing ---------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        self._writer.observe(transition)

    def should_update(self, frame: int) -> bool:
        if self.buffer.num_sampleable < self.cfg.batch_size:
            # Not a missed gradient step: these frames owe none, so the clock
            # skips past them rather than banking credit to spend in one burst.
            self._clock.skip(frame)
            return False
        self._clock.advance(frame)
        return self._clock.due

    # -- replica averaging -------------------------------------------------

    def named_parameters(self):
        """The online network's parameters, which are what replicas average.

        The target network is deliberately absent: it is a periodic copy of
        these, so averaging it as well would average a quantity that is about
        to be overwritten from the averaged online net at the next sync.
        """
        return self.online.named_parameters()

    def parameters(self):
        return self.online.parameters()

    # -- learning ----------------------------------------------------------

    def _build_step(self) -> TrainStep:
        inputs = step_inputs(self.obs_spec, self.device, batch=self.cfg.batch_size,
                             num_actions=self.num_actions, mask_n=self.mask_n,
                             prioritized=self.cfg.priority is not None)
        return TrainStep(self._step_fn, inputs, self.device, optimizer=self.optimizer,
                         mode=self.cfg.step_mode, cfg=self._engine)

    def _step_fn(self, inputs: dict) -> dict:
        return update_loss(self.online, self.target, inputs,
                           leaves=self.obs_spec.names,
                           num_actions=self.num_actions, mask_n=self.mask_n,
                           double=self.cfg.double, huber_delta=self.cfg.huber_delta,
                           prioritized=self.cfg.priority is not None)

    def update(self) -> dict[str, float]:
        """Every gradient step the frames collected so far are worth.

        One call rather than one step, because a vector env advancing several
        frames per round can be worth more than one step and the trainer asks
        once per round. The averaged metrics carry the count.
        """
        steps = max(1, self._clock.owed)
        totals: dict[str, float] = {}
        for _ in range(steps):
            for key, value in self._gradient_step().items():
                totals[key] = totals.get(key, 0.0) + value
        self._clock.consume(steps)
        metrics = {key: value / steps for key, value in totals.items()}
        metrics["gradient_steps"] = float(steps)
        metrics["epsilon"] = self.epsilon()
        return metrics

    def _gradient_step(self) -> dict[str, float]:
        batch = sample_for_update(self.buffer, self.cfg, self.update_count)
        step = self._step
        for leaf, spec in self.obs_spec.specs():
            step.load_input(leaf_key("obs", leaf), obs_rows(batch.obs[leaf], spec))
            step.load_input(leaf_key("next_obs", leaf),
                            obs_rows(batch.next_obs[leaf], spec))
        step.load_input("rewards", batch.rewards.reshape(-1, 1))
        step.load_input("dones", batch.dones.reshape(-1, 1))
        step.load_input("discounts", batch.discounts.reshape(-1, 1))
        onehot = np.zeros((len(batch), self.num_actions), np.float32)
        onehot[np.arange(len(batch)), batch.actions.reshape(-1).astype(np.int64)] = 1.0
        step.load_input("actions_onehot", onehot)
        if self.mask_n:
            step.load_input("next_mask", batch.next_mask)
        if batch.weights is not None:
            step.load_input("is_weights", batch.weights.reshape(-1, 1))

        out = step.run()
        if batch.weights is not None:
            write_back_priorities(self.buffer, batch, out.pop("td_abs"))
        self.update_count += 1
        if self.update_count % self.cfg.target_sync_interval == 0:
            # In-place copy: rebinding here would detach the target net from
            # any captured trace.
            self.target.copy_from(self.online)
        grad_sq = float(np.asarray(out.get("grad_sq", 0.0)).reshape(-1)[0])
        return {
            "loss": float(np.asarray(out["loss"]).reshape(-1)[0]),
            "mean_q": float(np.asarray(out["mean_q"]).reshape(-1)[0]),
            "grad_norm": float(np.sqrt(max(grad_sq, 0.0))),
        }

    # -- checkpointing -----------------------------------------------------

    #: The parameters, and the lagged copy of them the algorithm keeps. Both
    #: belong to the weights half: a target network restored from scratch
    #: would be a different bootstrap than the one that was checkpointed.
    _WEIGHT_KEYS = ("online", "target")
    _RUN_KEYS = ("optimizer", "frame_count", "update_count", "rng_state",
                 "autoreset", "clock")

    def state_dict(self, *, weights_only: bool = False) -> dict:
        return assemble(
            self.cfg,
            {"online": self.online.state_dict(), "target": self.target.state_dict()},
            {
                "optimizer": self.optimizer.state_dict(),
                "frame_count": self.frame_count,
                "update_count": self.update_count,
                "rng_state": self._rng.bit_generator.state,
                "autoreset": self._writer.state_dict(),
                "clock": self._clock.state_dict(),
            },
            weights_only=weights_only,
        )

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        load_run = validate(state, algo="DQN", config=self.cfg,
                            weight_keys=self._WEIGHT_KEYS, run_keys=self._RUN_KEYS,
                            weights_only=weights_only)
        self.online.load_state_dict(state["online"], self.device)
        self.target.load_state_dict(state["target"], self.device)
        if not load_run:
            return
        self.optimizer.load_state_dict(state["optimizer"], self.device)
        self.frame_count = int(state["frame_count"])
        self.update_count = int(state["update_count"])
        self._rng.bit_generator.state = state["rng_state"]
        self._writer.load_state_dict(state["autoreset"])
        self._clock.load_state_dict(state["clock"])

    def replay_state(self) -> dict:
        """The transitions this agent has collected. See `ttrl.buffers.persist`."""
        return replay_state(self.buffer)

    def load_replay_state(self, state: dict) -> None:
        load_replay_state(self.buffer, state)

    def close(self) -> None:
        """Release the captured trace and the staging buffers. Idempotent."""
        self._step.close()
        self._acting.close()
