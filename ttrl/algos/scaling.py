"""Per-replica budgets, and the network batch a vector env must fit.

Two things a trainer needs from an algorithm's config without knowing which
algorithm it is. Both exist because the alternative is the trainer guessing:
guessing which fields are per-run frame budgets, and guessing which field is
the batch the network was built for, which is `minibatch_size` for PPO and
`batch_size` for the off-policy pair.
"""

from __future__ import annotations

from dataclasses import fields, replace

__all__ = ["check_network_batch", "config_hash", "divide_for_replicas"]


def config_hash(config) -> int:
    """`hash()` for a frozen config that holds a keyword mapping.

    A frozen dataclass's generated `__hash__` hashes its field values, so one
    mapping field (`encoder_kwargs`) would make the whole config unhashable
    and stop it being a dict key or a set member, which a frozen value is
    expected to be. A mapping is hashed as its sorted items, which an
    encoder's keywords always are: a width, a depth, a flag.
    """
    parts = []
    for f in fields(config):
        value = getattr(config, f.name)
        parts.append(tuple(sorted(value.items())) if isinstance(value, dict)
                     else value)
    return hash((type(config).__name__, tuple(parts)))


def divide_for_replicas(config, n: int, names: tuple[str, ...]):
    """A copy of `config` with each named field divided by `n`.

    An n-chip run gives each replica 1/n of the frames, so every field that is
    a budget measured in frames (a replay capacity, a warm-up, an exploration
    schedule's horizon) has to be divided or it silently means something else
    at n chips than at one: a schedule written to anneal over the run would
    anneal n times slower, and the replay buffers would together hold n times
    the configured capacity.

    Raises on a field that does not divide cleanly. Truncating would make the
    realised budget a function of the chip count in a way nothing records.
    """
    if n < 1:
        raise ValueError(f"divide_for_replicas: n must be >= 1, got {n}")
    if n == 1:
        return config
    known = {f.name for f in fields(config)}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"{type(config).__name__}.scale_for_replicas: no such field(s) "
            f"{unknown}"
        )
    divided = {}
    for name in names:
        value = getattr(config, name)
        if value is None:
            continue
        if int(value) % n != 0:
            raise ValueError(
                f"{type(config).__name__}.scale_for_replicas({n}): {name}="
                f"{value} does not divide by {n} replicas. Each replica would "
                f"get {int(value) // n}, so the run's realised {name} would be "
                f"{(int(value) // n) * n} rather than {value}. Pick a value "
                f"divisible by the chip count."
            )
        divided[name] = int(value) // n
    return replace(config, **divided)


def check_network_batch(config, n_envs: int, algo: str, *,
                        field: str = "env.n_envs") -> None:
    """Refuse a lane count wider than the batch the networks are built for.

    One place computes this refusal, for every algorithm and for every width
    the acting path is asked for, because the reason is the same everywhere: a
    convolution fixes its batch at construction, so the acting forward runs
    through the same fixed-size buffer as the update and cannot be handed more
    lanes than that.

    `field` names the config field the count came from. There is more than one:
    a run acts at `env.n_envs` while it trains and at the eval env's own lane
    count while it scores, and a caller handed the wrong field name would go
    looking in the wrong place.
    """
    if n_envs < 1:
        raise ValueError(f"{algo}: {field} must be >= 1, got {n_envs}")
    batch = int(config.network_batch)
    if n_envs > batch:
        raise ValueError(
            f"{algo}: {field}={n_envs} exceeds the network batch of {batch} "
            f"({type(config).__name__}.{config.NETWORK_BATCH_FIELD}). A "
            f"convolution fixes its batch at construction, so the acting "
            f"forward runs through the same fixed-size buffer as the update. "
            f"Raise {config.NETWORK_BATCH_FIELD} to at least {n_envs}, or run "
            f"fewer environments."
        )
