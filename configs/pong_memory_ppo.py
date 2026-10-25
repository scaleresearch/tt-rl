"""Pong from a SINGLE frame, recurrent PPO. Memory is worth between 7.6 and
12.2 points of final score on this task.

This config exists to make the recurrent column evidence rather than a smoke
test, so the task is chosen to be one a feedforward policy cannot solve in
principle. `stack: 1` is the whole of it: one 84x84 greyscale frame shows
where the ball and both paddles ARE and says nothing about which way the ball
is travelling. Velocity is the state Pong is played on, so a memoryless
policy is guessing the sign of the ball's dx, while a GRU carrying the
previous frames' features can recover it.

The comparison that makes the number mean something is this same file with
memory switched off:

    ttrl train configs/pong_memory_ppo.py algo_cfg.recurrent=null

That control differs from this run in the GRU and in nothing else: same game,
same `stack: 1`, same rollout, same minibatch rows, same optimizer. A config
whose recurrent and feedforward scores match would be a config on a fully
observed task, which is why `stack` is 1 here and 4 in `pong_ppo.py`.

Measured at 1M frames, seed 0, 20 eval episodes, all three arms run from one
pinned export of commit c5056ae:

       frames   recurrent   feedforward   GRU-no-memory
      100,000      -21.00        -21.00          -21.00
      300,000      -19.75        -17.90          -21.00
      500,000      -15.65        -16.40          -20.35
      700,000      -11.45        -14.30          -17.00
    1,000,000       -2.30        -13.65          -18.85

Mean of the last three evals: -6.42, -14.05, -18.65.

Column 2 is the feedforward control above, the GRU removed. Column 3 keeps
the GRU and all 1,939,751 parameters and defeats the memory it carries.

Defeating the memory with the architecture and the parameter count held
fixed costs 12.2 points (column 3). Removing the GRU entirely costs 7.6
(column 2). The claim this file makes is the bracket between them rather
than either number alone, because the two controls differ in more than
memory: a severed GRU feeds its heads a 128-wide gated projection, while the
feedforward arm's heads read the full 512-wide encoder output, so column 3
is handicapped in representation width as well as in memory.

This supersedes an earlier 14.5-point figure taken from an unpinned 1.4M-frame
pair.

Every other number is `pong_ppo.py`'s, which is CleanRL's ppo_atari.py preset,
so that nothing but observability and memory separates this file from the
shipped Pong config.
"""

from __future__ import annotations

from ttrl import (DeviceConfig, EnvConfig, PPOConfig, RecurrentConfig,
                  TrainConfig)

GAME = "ALE/Pong-v5"

#: One frame per observation. The reason this config is not pong_ppo.py.
STACK = 1

config = TrainConfig(
    algo="ppo",
    total_frames=2_000_000,
    eval_interval=100_000,
    eval_episodes=10,
    checkpoint_interval=500_000,
    log_dir="runs/pong_memory_ppo",
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
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": True,
                "stack": STACK},
        seed=0,
    ),
    eval_env=EnvConfig(
        name="atari",
        n_envs=8,
        kwargs={"game": GAME, "sticky": 0.25, "episodic_life": False,
                "stack": STACK, "reward": "none"},
    ),
    algo_cfg=PPOConfig(
        n_steps=128,
        epochs=4,
        clip_eps=0.1,
        lam=0.95,
        lr=2.5e-4,
        anneal_lr=True,
        max_grad_norm=0.5,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        # pong_ppo.py's note applies unchanged: 128 rows is the largest batch
        # conv2's input gradient fits in L1 at. The recurrent update reshapes
        # those same 128 rows into 8 sequences of 16, so the convolution sees
        # the batch it was built for either way and this number is comparable
        # between the two runs.
        minibatch_size=128,
        # seq_len must divide n_steps (128) and minibatch_size (128). 16 steps
        # of truncated BPTT is 64 ALE frames at frameskip 4, which is several
        # ball crossings' worth of credit and still eight sequences per
        # minibatch, so the gradient is averaged over eight independent
        # (chunk, lane) draws rather than one long one.
        recurrent=RecurrentConfig(hidden_size=128, seq_len=16),
    ),
)
