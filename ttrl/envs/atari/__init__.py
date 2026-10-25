"""Atari, preprocessed the way the DQN literature does it.

Every Atari assumption in the framework lives in this package. The canonical
pipeline (Mnih et al. 2015, refined by Machado et al. 2018) is a list of
unglamorous details that together decide whether an agent learns at all:
greyscale at 84x84, pixel max over the last two raw frames, action repeat of
4, a stack of the last 4 processed frames, sticky actions, episodic life, and
reward sign clipping during training only. They are defaults of `AtariConfig`
here rather than constants anywhere else, so another suite never inherits
them and a user who wants 64x64 with a stack of 2 changes one field.

`cv2`, `ale_py` and `gymnasium` are imported at call time, so `import ttrl`
and `import ttrl.envs` work with no Atari extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ttrl.envs.atari.emulator import AtariEmulator, make_emulator
from ttrl.envs.atari.wrappers import EpisodicLife, FireReset, MaxAndSkip, NoopReset
from ttrl.envs.wrappers import (
    ClipReward,
    FrameStack,
    RecordEpisodeStatistics,
    ResizeObservation,
    SignReward,
)

__all__ = [
    "AtariConfig",
    "AtariEmulator",
    "EpisodicLife",
    "FireReset",
    "MaxAndSkip",
    "NoopReset",
    "make_atari",
]

RewardTransform = Literal["none", "sign", "clip"]


@dataclass(frozen=True)
class AtariConfig:
    """Every knob the Atari pipeline used to hardcode."""

    game: str = "ALE/Pong-v5"
    screen: int = 84
    stack: int = 4
    frameskip: int = 4
    noop_max: int = 30
    noop_action: int = 0
    sticky: float = 0.25
    #: Emulator frames one episode may take before the ALE truncates it. The
    #: ALE evaluation protocol (Machado et al. 2018) bounds an episode at
    #: 108,000 frames, which at `frameskip` 4 is 27,000 agent steps, and it
    #: bounds training episodes as well as scored ones. A truncation is an
    #: ordinary episode end: it is reported as `truncated`, the statistics
    #: wrapper emits the return, and an evaluation scores it like any other.
    #: Without it a deterministic policy that has found a long loop holds one
    #: episode open for as long as the game lets it, which an inline eval
    #: cannot wait out.
    max_frames: int = 108_000
    episodic_life: bool = True
    reward: RewardTransform = "sign"
    reward_clip: tuple[float, float] = (-1.0, 1.0)
    render_mode: str | None = None

    def __post_init__(self) -> None:
        for name in ("screen", "stack", "frameskip", "noop_max", "max_frames"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not 0.0 <= self.sticky <= 1.0:
            raise ValueError(
                f"sticky is repeat_action_probability and must be in [0, 1], "
                f"got {self.sticky}"
            )
        if self.reward not in ("none", "sign", "clip"):
            raise ValueError(
                f"reward must be one of none, sign, clip; got {self.reward!r}"
            )

    @classmethod
    def evaluation(cls, game: str, **kwargs) -> AtariConfig:
        """The ALE evaluation protocol: real scores, real episode boundaries.

        Episodic life moves episode boundaries to life losses, and any reward
        transform changes the number reported, so this sets both off. Passing
        either explicitly is a mistake rather than a preference, so it raises.
        """
        fixed = [k for k in ("episodic_life", "reward") if k in kwargs]
        if fixed:
            raise ValueError(
                f"AtariConfig.evaluation sets {fixed} itself and they cannot be "
                f"overridden: life-loss termination and reward transforms both "
                f"change the score the ALE protocol asks for."
            )
        return cls(game=game, episodic_life=False, reward="none", **kwargs)


def make_atari(
    game: str | None = None, *, config: AtariConfig | None = None, seed: int | None = None, **kwargs
):
    """Build a preprocessed Atari env yielding uint8 (stack, screen, screen) frames.

    Pass either a ready `AtariConfig` or field overrides as keywords.
    """
    if config is not None:
        if game is not None or kwargs:
            raise ValueError(
                "make_atari takes either config= or field overrides, not both"
            )
    else:
        if game is not None:
            kwargs["game"] = game
        config = AtariConfig(**kwargs)

    emulator = make_emulator(
        config.game, sticky=config.sticky, render_mode=config.render_mode,
        max_frames=config.max_frames,
    )
    # Statistics sit directly on the emulator so they record the real game
    # score across all lives, below both episodic life and any reward
    # transform. Episode length is therefore counted in raw ALE frames.
    env = RecordEpisodeStatistics(emulator)
    env = NoopReset(env, noop_max=config.noop_max, noop_action=config.noop_action)
    env = MaxAndSkip(env, skip=config.frameskip)
    if config.episodic_life:
        env = EpisodicLife(env, emulator)
    env = FireReset(env, emulator)
    env = ResizeObservation(env, config.screen, config.screen, greyscale=True)
    env = FrameStack(env, config.stack)
    if config.reward == "sign":
        env = SignReward(env)
    elif config.reward == "clip":
        env = ClipReward(env, *config.reward_clip)
    if seed is not None:
        env.reset(seed=seed)
    return env
