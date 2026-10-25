"""Breakout, clipped PPO. Measured: 125.2 at 5.5M frames, on one chip.

Measured on one chip over 5.5M frames: 10.0 at 250k climbing
near-monotonically to 125.2, which was its best and its last, with the
three closing evals at 100.0, 120.3 and 125.2. Three shallow dips on the
way, at 3.5M, 4.25-4.5M and 5.0M, no collapse and no stall. That run was
launched with `total_frames=5500000`, which also shortens the `anneal_lr`
horizon, so a 10M run anneals differently and is not the same run carried
on. The untrained baseline for this config is unmeasured.

Adding a game is adding one module like this one. The environment is the
registered name "atari" plus the keywords its builder takes, so nothing about
Atari is stated anywhere else in the run configuration.
"""

from __future__ import annotations

from ttrl import DeviceConfig, EnvConfig, PPOConfig, TrainConfig

GAME = "ALE/Breakout-v5"

config = TrainConfig(
    algo="ppo",
    total_frames=10_000_000,
    eval_interval=250_000,
    eval_episodes=30,
    checkpoint_interval=500_000,
    log_dir="runs/breakout_ppo",
    seed=0,
    device=DeviceConfig(
        # A physical chip id, the number in /dev/tenstorrent/<n> that
        # `ttrl doctor` prints. None takes the lowest chip nobody holds.
        chip=None,
        l1_small_size=32768,
        trace_region_size=33554432,
    ),
    env=EnvConfig(
        name="atari",
        n_envs=8,
        # CleanRL's ppo_atari.py wraps EpisodicLifeEnv, and every other number
        # in this file is its preset. Life loss is the dense early boundary
        # signal that truncates GAE on Breakout-likes, and dropping it alone
        # would define a different task from the preset the rest of this file
        # copies. The score is unaffected: eval_env below turns it off.
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": True},
        seed=0,
    ),
    eval_env=EnvConfig(
        name="atari",
        n_envs=8,
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": False,
                "reward": "none"},
    ),
    algo_cfg=PPOConfig(
        n_steps=128,
        epochs=4,
        clip_eps=0.1,
        lam=0.95,
        lr=2.5e-4,
        # The horizon anneal_lr divides by is left unset so that
        # TrainConfig.total_frames above fills it, which keeps one source for
        # the run's length. Stating it here too would mean two numbers to keep
        # in agreement for no gain.
        anneal_lr=True,
        max_grad_norm=0.5,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        # CleanRL splits the 8x128 rollout into 4 minibatches of 256. 256 does
        # not fit: conv2's input gradient exceeds L1 at that batch under every
        # placement (the forward is fine), so this is 8 minibatches of 128,
        # which is the largest power of two that builds. Raising it fails when
        # the agent is constructed, not mid-run.
        minibatch_size=128,
    ),
)
