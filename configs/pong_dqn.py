"""Pong, Double DQN. Measured: 17.8 at 10M frames, on one chip.

Measured over a 10M-frame run: -13.9 at 500k, -3.2 at 1M, 7.0 at 2M and 15.7
at 3M, then a plateau between 16 and 18 to the end, with its best of 17.8 at
7M. That best checkpoint scores 19.6 +/- 1.3 greedy over 30 episodes on CPU.

Adding a game is adding one module like this one. The environment is the
registered name "atari" plus the keywords its builder takes, so nothing about
Atari is stated anywhere else in the run configuration.
"""

from __future__ import annotations

from ttrl import DeviceConfig, DQNConfig, EnvConfig, TrainConfig

GAME = "ALE/Pong-v5"

config = TrainConfig(
    algo="dqn",
    total_frames=10_000_000,
    eval_interval=250_000,
    eval_episodes=30,
    checkpoint_interval=500_000,
    log_dir="runs/pong_dqn",
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
        n_envs=1,
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": True},
        seed=0,
    ),
    # The published number comes off the ALE evaluation protocol: whole-game
    # episodes and untransformed rewards. Training shapes both (life loss as
    # an episode boundary, reward sign clipping), so the scored environment is
    # stated separately rather than derived from the training one by a flag.
    eval_env=EnvConfig(
        name="atari",
        n_envs=1,
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": False,
                "reward": "none"},
    ),
    algo_cfg=DQNConfig(
        double=True,
        n_step=3,
        huber_delta=1.0,
        lr=1.0e-4,
        # One update every 4 frames: Mnih et al. 2015 replay ratio 0.25.
        train_freq=4,
        # Counted in UPDATES, not frames: 8000 updates is 32k frames at train_freq=4.
        target_sync_interval=8_000,
        buffer_capacity=1_000_000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_frames=1_000_000,
        batch_size=32,
        gamma=0.99,
        # Frames of pure collection before the first gradient update.
        # CleanRL dqn_atari.py uses 80_000; Mnih et al. 2015 uses 50_000 and
        # SB3's Atari zoo 100_000.
        learning_starts=80_000,
    ),
)
