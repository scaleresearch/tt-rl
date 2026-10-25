"""Hopper-v5, SAC. Measured on three seeds at 1M frames: 3604, 1857 and 3561.

The robotics column's shipped run, and the first config here that is not
Atari: a float32 observation of 11 numbers, 3 bounded torques, and a reward
with magnitude. Nothing about MuJoCo appears outside `env` below, the same way
nothing about ALE appears outside `env` in the Atari configs.

Measured on three seeds of this config, eval mean over 10 episodes, each on
one chip:

    seed   at 1M frames   best eval (frame)
    0      3604           3604 (1M)
    1      1857           2675 (600k)
    2      3561           3597 (975k)

Single evals swing widely between checkpoints on every seed, so
`checkpoint-best.npz` is the one to keep.
"""

from __future__ import annotations

from ttrl import DeviceConfig, EnvConfig, SACConfig, TrainConfig

TASK = "Hopper-v5"

config = TrainConfig(
    algo="sac",
    total_frames=1_000_000,
    eval_interval=25_000,
    eval_episodes=10,
    checkpoint_interval=250_000,
    log_dir="runs/hopper_sac",
    # env.seed is left unset and follows this one, so the run has one seed.
    seed=1,
    device=DeviceConfig(
        # A physical chip id, the number in /dev/tenstorrent/<n> that
        # `ttrl doctor` prints. None takes the lowest chip nobody holds.
        chip=None,
        l1_small_size=32768,
        trace_region_size=33554432,
    ),
    env=EnvConfig(
        name="mujoco",
        # One lane. SAC is off-policy and takes one gradient step per
        # environment step here, so the frame budget is the gradient budget
        # and extra lanes would spend frames without spending updates.
        n_envs=1,
        kwargs={"task": TASK},
    ),
    # No eval_env: Hopper's reward is the task's own on both paths, and the
    # episode bound is the one Gymnasium registers with the task, so there is
    # no training-side shaping for an evaluation to undo.
    algo_cfg=SACConfig(
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=200_000,
        learning_starts=5_000,
        train_freq=1,
        n_step=1,
        n_critics=2,
        learn_alpha=True,
        init_alpha=1.0,
        # None means -action_dim, which is -3 here: Haarnoja et al. 2018's
        # value for every continuous task.
        target_entropy=None,
        # The critic topology the runs above were measured with, PINNED
        # rather than inherited: the action fuses after the observation's own
        # 'mlp' tower and one hidden layer follows the concatenation. SAC's
        # default critic has since moved to the published input-fused topology
        # (critic_encoder 'auto' now means 'identity' on a vector leaf, with
        # critic_hidden=(256, 256)), so 'mlp' is named here outright:
        # inheriting a different critic would leave the numbers above
        # describing a network this config no longer builds.
        encoder="auto",
        critic_encoder="mlp",
        critic_hidden=(256,),
    ),
)
