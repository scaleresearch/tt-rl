"""Soft Actor-Critic: twin Q critics, a tanh-squashed Gaussian policy, a
learnable temperature and Polyak target updates.

The temperature is parameterised as log alpha so the optimizer works on an
unconstrained scalar and alpha stays positive by construction. Every loss
below is composed from primitive ops; no distribution primitive was added to
the engine for it.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.base import Transition, require_continuous
from ttrl.buffers.persist import load_replay_state, replay_state
from ttrl.algos.checkpoint import assemble, validate
from ttrl.algos.mask import mask_width
from ttrl.algos.clock import UpdateClock
from ttrl.algos.offpolicy import (BatchBuffers, ReplayWriter,
                                  sample_for_update, write_back_priorities)
from ttrl.algos.scaling import check_network_batch
from ttrl.algos.sac.config import SACConfig
from ttrl.algos.sac.losses import actor_loss, alpha_loss, critic_loss
from ttrl.algos.sac.policy import SquashedGaussian
from ttrl.algos.sac.step import (RegionOptimizers, actor_region,
                                 critic_region, summarize)
from ttrl.algos.sac.targets import PolyakSync
from ttrl.buffers import ReplayBuffer
from ttrl.engine.autograd import Tensor, no_grad
from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.ops import add, exp, mul
from ttrl.engine.optim import Adam
from ttrl.engine.step import TrainStep
from ttrl.engine.tensor import restore_in_place
from ttrl.nets.build import build_sac_nets
from ttrl.nets.acting import ActingNet, check_encoder_layout
from ttrl.spaces import ActionSpec, ObsGroup, require_group

__all__ = ["SAC"]


class SAC:
    """SAC over spec-built twin critics and a Gaussian policy. Continuous only."""

    def __init__(self, obs_spec: ObsGroup, action_spec: ActionSpec, device, *,
                 cfg: SACConfig = SACConfig(), engine: EngineConfig = DEFAULT,
                 n_envs: int = 1, seed: int | None = None, profile=None):
        require_continuous(action_spec, "sac")
        # Refuses a masked observation spec and names why: SAC's policy is a squashed
        # Gaussian over a continuous action vector, which has no finite set of
        # moves for a mask to pick out.
        mask_width(obs_spec, action_spec, "sac")
        check_network_batch(cfg, n_envs, "sac")
        self.obs_spec = require_group(obs_spec, "sac")
        self.action_spec = action_spec
        self.device = device
        self.profile = profile
        self.cfg = cfg
        self._engine = engine
        self.n_envs = int(n_envs)
        self.action_dim = int(action_spec.dim)

        if seed is not None:
            np.random.seed(seed)
        self.nets = build_sac_nets(obs_spec, action_spec, device,
                                   batch=cfg.batch_size, encoder=cfg.encoder,
                                   critic_encoder=cfg.critic_encoder,
                                   encoder_kwargs=cfg.encoder_kwargs,
                                   critic_encoder_kwargs=cfg.critic_encoder_kwargs,
                                   n_critics=cfg.n_critics,
                                   critic_hidden=cfg.critic_hidden)
        check_encoder_layout(self.nets.policy)
        self.dist = SquashedGaussian(action_spec, device)
        self.target_entropy = (float(-self.action_dim) if cfg.target_entropy is None
                               else float(cfg.target_entropy))
        # A (1, 1) tensor rather than a host float, so alpha stays on device
        # through the actor and critic losses and the update reads host memory
        # exactly once, for the metric.
        self.log_alpha = Tensor.from_numpy(
            np.full((1, 1), np.log(cfg.init_alpha), np.float32), device,
            requires_grad=True)

        self.actor_opt = Adam(self.nets.policy_parameters(), lr=cfg.lr)
        self.critic_opt = Adam(self.nets.critic_parameters(), lr=cfg.lr)
        self.alpha_opt = Adam([self.log_alpha], lr=cfg.lr)

        self.buffer = ReplayBuffer(
            obs_spec, action_spec, capacity=cfg.buffer_capacity, n_envs=self.n_envs,
            n_step=cfg.n_step, gamma=cfg.gamma, seed=seed,
            priority=cfg.priority, sampler=cfg.sampler,
        )
        self._writer = ReplayWriter(self.buffer, self.n_envs)
        self._clock = UpdateClock(cfg.train_freq, cfg.learning_starts)
        # Acting runs at the lane count; batch_size is a replay-ratio knob.
        self._acting = ActingNet(self.nets.policy, obs_spec, device,
                                 width=cfg.batch_size, lanes=self.n_envs)
        noise = (cfg.batch_size, self.action_dim)
        extra = {"noise": noise, "next_noise": noise}
        if cfg.priority is not None:
            extra["is_weights"] = (cfg.batch_size, 1)
        self._batch = BatchBuffers(obs_spec, action_spec, device, batch=cfg.batch_size,
                                   extra=extra)
        self._build_steps()
        self.frame_count = 0
        self.update_count = 0
        self._rng = np.random.default_rng(seed)
        self._eval_rng = np.random.default_rng(seed).spawn(1)[0]

    # -- acting ------------------------------------------------------------

    def act(self, obs, *, deterministic: bool = False,
            episode_starts=None) -> np.ndarray:
        """One action per lane. `episode_starts` is part of the acting
        contract for a policy with memory; this one has none to reset."""
        observation = self.obs_spec.observation(obs)
        batch = int(observation.lanes)
        with no_grad():
            mean_t, log_std_t = self._acting(observation)
        mean = mean_t.numpy()[:batch]
        log_std = log_std_t.numpy()[:batch]
        rng = self._eval_rng if deterministic else self._rng
        action, _logp = self.dist.act(mean, log_std, rng, deterministic)
        if not deterministic:
            self.frame_count += batch
        return action

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
        """The policy and the critics, and the temperature.

        The Polyak targets are absent: they are a lagged average of the
        critics, so hard-copying an averaged critic into them would destroy
        exactly the lag SAC's target depends on, and they reach the averaged
        critics on their own within a few hundred updates. `log_alpha` is
        included because it is a trained parameter whose value changes what
        every replica optimises, not a local accumulator.
        """
        named = [(f"policy.{n}", p) for n, p in self.nets.policy.named_parameters()]
        named += [(f"critics.{n}", p) for n, p in self.nets.critics.named_parameters()]
        return iter(named + [("log_alpha", self.log_alpha)])

    def parameters(self):
        return (param for _name, param in self.named_parameters())

    # -- learning ----------------------------------------------------------

    def alpha(self) -> Tensor:
        """The temperature as a detached (1, 1) device tensor."""
        with no_grad():
            return exp(self.log_alpha)

    def _critic_loss(self, alpha: Tensor):
        weights = self._batch["is_weights"] if self.cfg.priority is not None else None
        return critic_loss(self.nets, self.dist, self._batch, alpha, weights)

    def _actor_loss(self, alpha: Tensor):
        return actor_loss(self.nets, self.dist, self._batch, alpha)

    def _alpha_loss(self, logp: Tensor):
        return alpha_loss(self.log_alpha, logp, self.target_entropy)

    def _build_steps(self) -> None:
        """One TrainStep per region. `inputs` is the batch staging, which
        SAC loads through `BatchBuffers` because its observations are
        `ObsPair` views rather than plain named tensors; both are allocated
        once and only written into, which is what a captured trace needs."""
        every = (self.actor_opt, self.critic_opt, self.alpha_opt)
        stepped = ([self.actor_opt, self.alpha_opt] if self.cfg.learn_alpha
                   else [self.actor_opt])
        prioritized = self.cfg.priority is not None

        def critic_fn(_inputs: dict) -> dict:
            return critic_region(self.nets, self.dist, self._batch,
                                 self.log_alpha, prioritized=prioritized)

        def actor_fn(_inputs: dict) -> dict:
            return actor_region(self.nets, self.dist, self._batch,
                                self.log_alpha, self.target_entropy,
                                learn_alpha=self.cfg.learn_alpha)

        def make(fn, stepped_opts):
            return TrainStep(fn, self._batch.tensors, self.device,
                             optimizer=RegionOptimizers(every, stepped_opts),
                             mode=self.cfg.step_mode, cfg=self._engine)

        self._critic_step = make(critic_fn, [self.critic_opt])
        self._actor_step = make(actor_fn, stepped)
        self._polyak = PolyakSync(self.nets, self.cfg.tau, self.device,
                                  traced=self.cfg.step_mode != "eager",
                                  cfg=self._engine)

    def update(self) -> dict[str, float]:
        """Every gradient step the frames collected so far are worth. See
        `ttrl.algos.clock.UpdateClock`."""
        steps = max(1, self._clock.owed)
        totals: dict[str, float] = {}
        for _ in range(steps):
            for key, value in self._gradient_step().items():
                totals[key] = totals.get(key, 0.0) + value
        self._clock.consume(steps)
        metrics = {key: value / steps for key, value in totals.items()}
        metrics["gradient_steps"] = float(steps)
        return metrics

    def _gradient_step(self) -> dict[str, float]:
        batch = sample_for_update(self.buffer, self.cfg, self.update_count)
        rows = len(batch)
        self._batch.load_batch(batch)
        self._batch.load("actions", batch.actions.reshape(rows, self.action_dim))
        self._batch.load("noise", self._rng.standard_normal((rows, self.action_dim)))
        self._batch.load("next_noise",
                         self._rng.standard_normal((rows, self.action_dim)))
        if batch.weights is not None:
            self._batch.load("is_weights", batch.weights.reshape(rows, 1))

        critic = self._critic_step.run()
        if batch.weights is not None:
            # Host-side numpy into the priority table. It runs before the next
            # `observe`, so no column has been overwritten between the draw
            # and this.
            write_back_priorities(self.buffer, batch, critic.pop("td_abs"))
        actor = self._actor_step.run()
        self._polyak.run()
        self.update_count += 1
        return summarize(critic, actor, learn_alpha=self.cfg.learn_alpha)

    # -- checkpointing -----------------------------------------------------

    #: `nets` carries the policy, the critics and the lagged targets; the
    #: targets are state rather than derived data, since a resume that rebuilt
    #: them from the critics would discard however far they had lagged.
    _WEIGHT_KEYS = ("nets", "log_alpha")
    _RUN_KEYS = ("actor_opt", "critic_opt", "alpha_opt", "frame_count",
                 "update_count", "rng_state", "autoreset", "clock")

    def state_dict(self, *, weights_only: bool = False) -> dict:
        return assemble(
            self.cfg,
            {"nets": self.nets.state_dict(), "log_alpha": self.log_alpha.numpy()},
            {
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "frame_count": self.frame_count,
                "update_count": self.update_count,
                "rng_state": self._rng.bit_generator.state,
                "autoreset": self._writer.state_dict(),
                "clock": self._clock.state_dict(),
            },
            weights_only=weights_only,
        )

    def load_state_dict(self, state: dict, *, weights_only: bool = False) -> None:
        load_run = validate(state, algo="SAC", config=self.cfg,
                            weight_keys=self._WEIGHT_KEYS, run_keys=self._RUN_KEYS,
                            weights_only=weights_only)
        self.nets.load_state_dict(state["nets"])
        restore_in_place(self.log_alpha.data, state["log_alpha"])
        if not load_run:
            return
        self.actor_opt.load_state_dict(state["actor_opt"], self.device)
        self.critic_opt.load_state_dict(state["critic_opt"], self.device)
        self.alpha_opt.load_state_dict(state["alpha_opt"], self.device)
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
        """Release the captured traces and the staging buffers. Idempotent."""
        self._critic_step.close()
        self._actor_step.close()
        self._polyak.close()
        self._acting.close()
        self._batch.close()
