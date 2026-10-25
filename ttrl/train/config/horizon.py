"""The one field this layer reaches into an algorithm's own config for.

A run's length is `TrainConfig.total_frames`, and an algorithm whose learning
rate decays over the run carries the same number in its own config. Reaching
across that boundary is a seam, so it is one module: the mirror of
`ttrl.train.config.env.align_reward_discount`, which does the same for the
discount. `ttrl.train.config.schema` applies it and holds nothing else about
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["DERIVED_MARKER", "RUN_HORIZON_FIELD", "align_run_horizon"]


#: The one field name this layer reaches into an algorithm's config for. A
#: schedule that runs to the end of training needs the run's length, and the
#: run's length is `TrainConfig.total_frames`.
RUN_HORIZON_FIELD = "total_frames"

#: Attribute this module sets on an algo config whose horizon it filled in.
#: It travels with the object, so a `TrainConfig` rebuilt from a loaded one
#: (`dataclasses.replace(cfg, total_frames=...)`, the way a variant of a
#: shipped config is written) knows the number it is carrying was derived
#: from the old horizon and re-derives it instead of refusing a
#: disagreement. A horizon the user typed carries no marker and is honoured.
DERIVED_MARKER = "_run_horizon_derived"


def align_run_horizon(algo_cfg: Any, total_frames: int) -> tuple[Any, bool]:
    """Give the algorithm's config the run's horizon, or refuse a disagreement.

    Returns it with whether the horizon was DERIVED, which `load_config`
    reads to re-open the follow under an override.

    An algorithm whose learning rate decays to zero over the run needs to know
    how long the run is, so its config carries the same number. Two fields for
    one quantity drift: a user who raises `TrainConfig.total_frames` and
    forgets the other gets a schedule that finishes early, which is a wrong
    run that looks like a healthy one. There is therefore one source. An
    unset field is filled from it; a field that was set to something else is
    refused rather than overwritten, because overwriting a value the user
    typed is its own surprise.

    A config whose horizon THIS function filled in carries `DERIVED_MARKER`,
    and such a value is re-derived rather than refused: deriving a variant of
    a shipped config with `dataclasses.replace(cfg, total_frames=...)` would
    otherwise collide with the horizon the shipped file was resolved at.

    Only a config that declares the field is touched, so an algorithm with no
    horizon-dependent schedule is left exactly as it was.
    """
    if algo_cfg is None or not dataclasses.is_dataclass(algo_cfg):
        return algo_cfg, False
    names = {f.name for f in dataclasses.fields(algo_cfg)}
    if RUN_HORIZON_FIELD not in names:
        return algo_cfg, False
    stated = getattr(algo_cfg, RUN_HORIZON_FIELD)
    if stated is None or getattr(algo_cfg, DERIVED_MARKER, False):
        replaced = dataclasses.replace(algo_cfg, **{RUN_HORIZON_FIELD: total_frames})
        object.__setattr__(replaced, DERIVED_MARKER, True)
        return replaced, True
    if int(stated) != int(total_frames):
        raise ValueError(
            f"{type(algo_cfg).__name__}.{RUN_HORIZON_FIELD} is {stated} but "
            f"TrainConfig.total_frames is {total_frames}. They are one quantity, "
            f"the length of this run, and a schedule keyed off the stale one "
            f"finishes at the wrong frame. Leave "
            f"{type(algo_cfg).__name__}.{RUN_HORIZON_FIELD} unset and it is "
            f"filled from TrainConfig.total_frames."
        )
    return algo_cfg, False
