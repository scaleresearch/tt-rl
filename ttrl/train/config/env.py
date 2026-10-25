"""The environment half of a run configuration.

Which environment to build, how many lanes, with what keywords, and the
normalisation applied ABOVE the vector environment. The last of those is here
rather than among the builder's keywords because each normaliser holds ONE
running estimator fitted on every lane's rows: a lane's builder can only make
a per-lane copy, which sees 1/N of the data and whitens each lane against its
own statistics. `ttrl.train.builders.build_env` applies them, and
`ttrl.envs.vec_normalize` is what it applies.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DISCOUNT_FIELD", "EnvConfig", "align_reward_discount"]

#: The field an algorithm's config calls its discount. Reward normalisation
#: divides by the standard deviation of the DISCOUNTED return, so the discount
#: it accumulates with and the one the critic regresses under are one quantity.
DISCOUNT_FIELD = "gamma"


@dataclass
class EnvConfig:
    """Which environment to build, how many lanes, and with what keywords.

    `name` is a key in `ttrl.envs.registry`; `kwargs` are that builder's own
    keywords, so adding an environment suite never adds a field here. An
    override reaches into them by path, for example `env.kwargs.sticky=0.1`.
    """

    name: str = "atari"
    n_envs: int = 1
    asynchronous: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    #: Whiten the observation with one running mean and variance shared by
    #: every lane (`ttrl.envs.vec_normalize.NormalizeObservation`, applied
    #: above the vector env by `build_env`). Stated here rather than as a
    #: builder keyword because the estimator is one object over all the
    #: lanes, which is not something a per-lane builder can make.
    normalize_obs: bool = False
    normalize_obs_clip: float = 10.0

    #: Divide the reward by the running standard deviation of the discounted
    #: return (`ttrl.envs.vec_normalize.NormalizeReward`). The environment's
    #: own undiscounted return is unaffected and is still what an eval
    #: reports: see that class and `ttrl.train.evaluate`.
    normalize_reward: bool = False
    normalize_reward_clip: float = 10.0
    #: The discount the return is accumulated with. One quantity with the
    #: algorithm's gamma, so it is left unset and filled from `algo_cfg.gamma`
    #: by `TrainConfig.__post_init__`, which refuses a disagreement rather
    #: than normalising against a discount the critic does not use. See
    #: `_align_reward_discount`.
    normalize_reward_gamma: float | None = None
    # The training env's own seed stream, kept separate from `TrainConfig.seed`
    # (agent init, sampling, eval) so a sweep can vary one without the other.
    # None means "follow TrainConfig.seed", which `TrainConfig.__post_init__`
    # resolves to an int: without that, `seed=7` alone would leave the
    # training trajectories on seed 0.
    seed: int | None = None

    #: True when the field below was left unset and therefore follows the
    #: run: `seed` follows `TrainConfig.seed`, `normalize_reward_gamma`
    #: follows `algo_cfg.gamma`. Neither is a dataclass field: both are
    #: derived state, which `config_to_dict` must not emit. `load_config`
    #: reads them to re-open each follow before overrides, so an override of
    #: the source moves the derived value with it.
    _seed_follows_run_seed = False  # NOT annotated: must stay out of fields()
    _reward_gamma_follows_algo = False  # NOT annotated, same reason

    def __post_init__(self) -> None:
        self._seed_follows_run_seed = self.seed is None
        self._reward_gamma_follows_algo = self.normalize_reward_gamma is None
        if not self.name:
            raise ValueError(
                "env.name is empty and must name a registered environment. "
                "The names this build ships are 'atari' and 'mujoco'; "
                "ttrl.ENVS.names() lists every registered one, and "
                "ttrl.envs.register(name, builder) adds yours."
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0 or None, got {self.seed}")
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {self.n_envs}")
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"env.kwargs must be a dict of builder keywords, got "
                f"{type(self.kwargs).__name__}"
            )
        for name in ("normalize_obs_clip", "normalize_reward_clip"):
            if not getattr(self, name) > 0.0:
                raise ValueError(
                    f"env.{name} must be positive, got {getattr(self, name)}")
        if (self.normalize_reward_gamma is not None
                and not 0.0 < self.normalize_reward_gamma <= 1.0):
            raise ValueError(
                f"env.normalize_reward_gamma must be in (0, 1] or None, got "
                f"{self.normalize_reward_gamma}")


def align_reward_discount(env, algo_cfg: Any, algo: str) -> None:
    """Give the env's reward normaliser the algorithm's discount, or refuse.

    Same argument as `_align_run_horizon`, for the same reason: two fields for
    one quantity drift, and a normaliser dividing by the deviation of a return
    discounted at 0.99 while the critic regresses one discounted at 0.95 is a
    wrong run that looks like a healthy one. There is therefore one source.

    Nothing is filled when reward normalisation is off, so a run that does not
    ask for it is never made to declare a discount, and an algorithm whose
    config has no discount is refused BY NAME when it is asked for, rather
    than defaulted to a number this layer invented.
    """
    if not env.normalize_reward:
        return
    stated = env.normalize_reward_gamma
    available = (
        algo_cfg is not None and dataclasses.is_dataclass(algo_cfg)
        and DISCOUNT_FIELD in {f.name for f in dataclasses.fields(algo_cfg)}
    )
    if not available:
        if stated is None:
            raise ValueError(
                f"env.normalize_reward is on and no discount is available for "
                f"it: {algo!r}'s config "
                f"{'is None' if algo_cfg is None else type(algo_cfg).__name__}"
                f" declares no {DISCOUNT_FIELD!r} field. Reward normalisation "
                f"divides by the deviation of the DISCOUNTED return, so there "
                f"is no such thing as doing it without a discount. State "
                f"env.normalize_reward_gamma, or pass an algo_cfg that "
                f"declares {DISCOUNT_FIELD}."
            )
        return
    theirs = float(getattr(algo_cfg, DISCOUNT_FIELD))
    if stated is None:
        env.normalize_reward_gamma = theirs
        return
    if float(stated) != theirs:
        raise ValueError(
            f"env.normalize_reward_gamma is {stated} but "
            f"{type(algo_cfg).__name__}.{DISCOUNT_FIELD} is {theirs}. They "
            f"are one quantity, the discount the return is taken under, and a "
            f"reward divided by the deviation of a differently discounted "
            f"return is scaled by a number the critic never sees. Leave "
            f"env.normalize_reward_gamma unset and it is filled from "
            f"{type(algo_cfg).__name__}.{DISCOUNT_FIELD}."
        )
