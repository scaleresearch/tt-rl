"""The two halves of an agent's state, and the rules for loading either.

A checkpoint carries a WEIGHTS half (the parameters, and the lagged copies of
them an algorithm keeps) and a RUN half (optimizer moments, frame and update
counts, the exploration RNG, the autoreset phase, the update clock). The
weights alone are a snapshot to score or to start from; the run half is what
makes a resume continue the same run rather than a similar one.

`state_dict(weights_only=True)` writes the weights half and marks it, so a
loader can tell the two apart instead of inferring it from which keys happen
to be present. Handing a marked state to a plain `load_state_dict` raises,
because the alternative is a resume that silently continues with zeroed Adam
moments, a restarted epsilon schedule and a lost frame count.
"""

from __future__ import annotations

import dataclasses

__all__ = ["CONFIG_KEY", "WEIGHTS_ONLY_KEY", "assemble", "describe_config_diff",
           "validate"]

#: Marks a state that carries the weights half only.
WEIGHTS_ONLY_KEY = "weights_only"

#: The algorithm's own config, present in both halves: loading parameters
#: trained under different physics is as wrong as resuming under them.
CONFIG_KEY = "config"


def assemble(config, weights: dict, run: dict, *, weights_only: bool) -> dict:
    """One state dict from its two halves."""
    state = dict(weights)
    state[CONFIG_KEY] = config
    if weights_only:
        state[WEIGHTS_ONLY_KEY] = True
        return state
    state.update(run)
    return state


def describe_config_diff(saved, current) -> str:
    """The differing fields and their two values, as one line each side.

    Printing both configs in full makes the reader diff every field by eye,
    which for a resume with a changed horizon is seventeen fields of noise
    around the one that matters. Every differing field is named and none is
    truncated: a truncated diff is the thing this replaces.
    """
    if not (dataclasses.is_dataclass(saved) and dataclasses.is_dataclass(current)
            and type(saved) is type(current)):
        # Different types entirely, or something that is not a config: there is
        # no field correspondence to report, so fall back to the whole objects.
        return f"checkpoint holds {saved!r}; this run has {current!r}"
    names = [f.name for f in dataclasses.fields(saved)
             if getattr(saved, f.name) != getattr(current, f.name)]
    shape = [n for n in names if n in getattr(current, "SHAPE_FIELDS", ())]
    was = ", ".join(f"{n}={getattr(saved, n)!r}" for n in names)
    now = ", ".join(f"{n}={getattr(current, n)!r}" for n in names)
    one = len(names) == 1
    head = f"{len(names)} field{'' if one else 's'} differ{'s' if one else ''}: "
    text = f"{head}checkpoint was written with {was}; this run has {now}"
    if shape:
        verb = "decides" if len(shape) == 1 else "decide"
        text += (
            f". {', '.join(shape)} {verb} the networks' shapes, so the saved "
            f"parameters do not describe this agent's networks at all"
        )
    return text


def validate(state: dict, *, algo: str, config, weight_keys: tuple[str, ...],
             run_keys: tuple[str, ...], weights_only: bool) -> bool:
    """Check a state against what `algo` needs, and say whether to load the run
    half from it.

    Returns False when the state carries no run half, or when the caller asked
    for the weights alone out of a full checkpoint, which is how an evaluation
    loads a resumable file without inheriting its optimizer.
    """
    carries_weights_only = bool(state.get(WEIGHTS_ONLY_KEY, False))
    if carries_weights_only and not weights_only:
        raise ValueError(
            f"{algo}.load_state_dict: this checkpoint carries parameters only; it "
            f"was written with state_dict(weights_only=True) and holds none of "
            f"{list(run_keys)}. Continuing from it would restart the optimizer "
            f"moments at zero, restart every schedule that reads a frame count, "
            f"and lose the exploration RNG, which is a different run rather than "
            f"this one continued. Pass weights_only=True to load the parameters "
            f"on purpose, or resume from a full checkpoint."
        )
    required = weight_keys + (CONFIG_KEY,)
    if not carries_weights_only:
        required = required + run_keys
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            f"{algo}.load_state_dict: checkpoint is missing {missing}. Each one is "
            f"needed to continue the checkpointed run rather than start a similar "
            f"one: the weights, the optimizer moments, the counts every schedule "
            f"reads, the sampling RNG, the autoreset phase and the configuration "
            f"all decide what happens next."
        )
    if state[CONFIG_KEY] != config:
        raise ValueError(
            f"{algo}.load_state_dict: this agent's configuration differs from "
            f"the checkpoint's. "
            f"{describe_config_diff(state[CONFIG_KEY], config)}. Loading would "
            f"change the algorithm mid-run; rebuild the agent with the "
            f"checkpoint's value for each field named above, or start a new run."
        )
    return not (carries_weights_only or weights_only)
