"""Watch a trained DQN checkpoint play, on screen and/or into an mp4.

    uv run python play.py CHECKPOINT --chip 2 --video pong.mp4
    uv run python play.py CHECKPOINT --chip 2 --window        # needs a display

The frame filmed is the emulator's native 160x210 colour frame, magnified
with nearest-neighbour, with the readout in a panel beside the playfield:
`ttrl.train.film.Film` composites it. The 84x84 greyscale stack the policy
trains on is unchanged and is only read, to show in the panel what the
agent actually sees.

Rendering is an optional capability rather than part of the `Env` protocol,
because most environments cannot produce a frame. `supports_render(env)` is
how a caller asks, and it is False for an environment built without
`render_mode="rgb_array"`. Vector environments do not render: compositing
lanes is the caller's problem, so this builds a single env to watch.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

import ttrl
import ttrl.envs
from ttrl.train.film import Film


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", default="",
                   help="checkpoint written by train.py (omit with --random)")
    p.add_argument("--chip", type=int, default=0, help="physical chip id")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--video", default="", help="write an mp4 here")
    p.add_argument("--window", action="store_true",
                   help="show a window (needs DISPLAY)")
    p.add_argument("--fps", type=int, default=15, help="15 = real time at frameskip 4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game", default="ALE/Pong-v5",
                   help="only used with --random; otherwise the checkpoint says")
    p.add_argument("--max-steps", type=int, default=6_000, help="per episode")
    p.add_argument("--random", action="store_true",
                   help="play uniformly at random instead of loading the "
                        "checkpoint. Opens no chip. This is the baseline to "
                        "compare a trained clip against.")
    return p.parse_args()


class RandomPolicy:
    """The baseline. Takes the action spec the environment reports."""

    def __init__(self, action_spec, seed):
        self.n = action_spec.n
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=False):
        return self.rng.integers(0, self.n, size=obs.shape[0])

    def close(self):
        return None


class Window:
    def __init__(self, size):
        # Deferred: pygame is only needed by --window.
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("ttrl: trained policy")
        self.clock = pygame.time.Clock()

    def show(self, rgb, fps):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
        surface = self.pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        self.pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        self.pygame.quit()


def require_render(env):
    """Refuse early rather than at the first frame."""
    if not ttrl.envs.supports_render(env):
        raise SystemExit(
            f"{type(env).__name__} cannot render. Build it with "
            'render_mode="rgb_array".')


def roll(env, agent, args, film):
    """Play `args.episodes` episodes, recording and/or showing each frame."""
    # Deferred: cv2 is only needed by --video.
    import cv2

    writer = window = None
    returns = []
    try:
        for episode in range(args.episodes):
            obs, _info = env.reset(seed=args.seed + episode)
            ret, step, action = 0.0, 0, 0
            while True:
                shot = film.frame(env.render(), ret=ret, step=step,
                                  action=action, n_actions=env.action_spec.n,
                                  agent_view=obs[-1])
                if args.video:
                    if writer is None:
                        h, w = shot.shape[:2]
                        writer = cv2.VideoWriter(
                            args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.fps), (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"could not open {args.video} for writing")
                    writer.write(shot[:, :, ::-1])  # cv2 wants BGR
                if args.window:
                    if window is None:
                        window = Window((shot.shape[1], shot.shape[0]))
                    if not window.show(shot, args.fps):
                        return returns

                # `act` takes a leading lane axis, as the training loop hands
                # it, so one env still arrives batched.
                action = int(agent.act(obs[None], deterministic=True)[0])
                obs, reward, terminated, truncated, _info = env.step(action)
                ret += float(reward)
                step += 1
                if terminated or truncated or step >= args.max_steps:
                    break
            returns.append(ret)
            print(f"  episode {episode + 1}: return {ret:+.0f} in {step} steps",
                  flush=True)
    finally:
        if writer is not None:
            writer.release()
            print(f"wrote {args.video}")
        if window is not None:
            window.close()
    return returns


def main():
    args = parse_args()
    if not args.random and not args.checkpoint:
        raise SystemExit("give a checkpoint path, or --random for the baseline")

    if args.random:
        # No chip, no checkpoint: just the emulator and the action spec.
        env = ttrl.envs.make_atari(
            game=args.game, episodic_life=False, reward="none",
            render_mode="rgb_array")
        require_render(env)
        print(f"{args.game} with a uniform-random policy (no chip opened)")
        try:
            film = Film(title=args.game, subtitle="uniform-random baseline",
                        action_names=env.unwrapped.action_meanings())
            returns = roll(env, RandomPolicy(env.action_spec, args.seed), args,
                           film)
        finally:
            env.close()
    else:
        cfg = ttrl.config_from_checkpoint(args.checkpoint)
        # The run's eval environment: whole-game episodes and untransformed
        # rewards, which is the score a viewer expects to see on screen.
        env_kwargs = dict(cfg.eval_env_config.kwargs)
        game = env_kwargs.get("game", args.game)

        # One call for the chip: it scopes this process before ttnn loads and
        # then opens the chip, which is the order a script must not get wrong.
        # The checkpoint's own device settings, with the chip asked for here.
        device_config = dataclasses.replace(cfg.device, chip=args.chip)
        with ttrl.device_for(device_config) as (dev, prof):
            env = ttrl.envs.make_atari(**env_kwargs, render_mode="rgb_array")
            require_render(env)
            agent, _cfg, loaded = ttrl.agent_from_checkpoint(
                args.checkpoint, dev, profile=prof, weights_only=True)
            print(f"{game} from {Path(args.checkpoint).name} "
                  f"(trained {loaded.get('frame', 0):,} frames)")
            try:
                film = Film(
                    title=game,
                    subtitle=f"trained {loaded.get('frame', 0):,} frames",
                    action_names=env.unwrapped.action_meanings())
                returns = roll(env, agent, args, film)
            finally:
                agent.close()
                env.close()

    if returns:
        print(f"mean return {np.mean(returns):+.2f} over {len(returns)} episode(s)")


if __name__ == "__main__":
    main()
