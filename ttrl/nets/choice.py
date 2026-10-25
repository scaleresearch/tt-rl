"""Which encoder builds one leaf's tower: a name, or a class handed over.

`ENCODERS` maps a NAME to a builder, and a name is what a config carries into
a checkpoint and back. What it could not carry was the other thing every
comparable framework accepts: the class itself, which is what trying a new
architecture looks like before it is worth naming and registering. This
module is the whole of that, kept out of `ttrl.nets.registry` so the rule
stays readable and out of `ttrl.nets.multi` so it can be exercised with no
chip: everything here imports the registry and nothing else.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from ttrl.nets.registry import ENCODERS

__all__ = ["UnnamedEncoder", "coerce_encoder", "resolve_encoder"]


@dataclasses.dataclass(frozen=True)
class UnnamedEncoder:
    """An encoder builder a run was handed directly, under no registered name.

    Every comparable framework takes the class as well as the name, because
    handing over a class is what trying a new architecture looks like before
    it is worth naming. A `str` field alone could not carry one, and putting
    the class in the field made the config unwritable: a checkpoint stores
    plain values, and the first thing a run did with the class was fail to
    serialize it hours in.

    WHAT THIS CANNOT PROMISE, said out loud instead of papered over. `target`
    is the class's own `module:qualname`, stored in the checkpoint so the file
    records which architecture trained those weights. It is NOT a way to
    rebuild it: nothing imports a module a checkpoint names, and no registry
    entry claims the name. So `builder` is not stored (`UNSAVED_FIELDS`), and
    a config read back out of a file carries the identity with no callable.
    `resolve` refuses that state by name, which makes a resume either
    re-supply the same class or register it. The registry is not weakened to
    buy this: a NAME still round-trips through a file and still resolves on
    its own, and nothing is auto-registered behind the user's back, which
    would make which builder a name means depend on import order.

    `builder` is out of the comparison too, so a resume that passes the same
    class again compares equal to the config in the file and continues the
    run rather than reporting a changed architecture.
    """

    target: str
    builder: Callable | None = dataclasses.field(
        default=None, compare=False, repr=False)

    #: `ttrl.train.checkpoint.tree` neither writes nor reads this field.
    UNSAVED_FIELDS = ("builder",)

    #: One config VALUE, so `ttrl.train.config.config_to_dict` keeps it whole.
    #: Decomposed into its fields it came back out of `coerce_encoder` as a
    #: per-leaf encoder mapping, and its callable reached `json.dumps`.
    CONFIG_VALUE = True

    def provenance(self) -> dict[str, str]:
        """How a run's `config.json` records this choice.

        The qualified name under its own key, never under `encoder` as if it
        were a registered name: nothing can resolve it, and a reader of the
        file must not be able to mistake it for something that can. See
        `ttrl.train.provenance`.
        """
        return {"unnamed_encoder": self.target}

    def resolve(self) -> Callable:
        if self.builder is None:
            raise ValueError(
                f"this run's encoder was handed over as the class "
                f"{self.target!r}, which no registered name identifies. The "
                f"checkpoint recorded which class it was, not a way to build "
                f"it: ttrl never imports a module a checkpoint names, and no "
                f"registry entry claims that name. Pass the same class again "
                f"as encoder=<the class> to continue this run, or give it a "
                f"name so the file alone is enough: "
                f"ttrl.ENCODERS.register('my_tower', MyEncoder) and "
                f"encoder='my_tower'."
            )
        return self.builder


def coerce_encoder(encoder: Any) -> Any:
    """An `encoder` config field as a name, a per-leaf mapping or an `UnnamedEncoder`.

    THE one place a handed-over class becomes a config value, called from
    every algorithm config's `__post_init__`, so `encoder=MyEncoder` works
    the same way whichever algorithm reads it and the stored config is always
    made of plain values.
    """
    if isinstance(encoder, dict):
        return {leaf: coerce_encoder(value) for leaf, value in encoder.items()}
    if isinstance(encoder, (str, UnnamedEncoder)):
        return encoder
    if callable(encoder):
        return UnnamedEncoder(
            target=f"{encoder.__module__}:{getattr(encoder, '__qualname__', encoder)}",
            builder=encoder,
        )
    raise TypeError(
        f"encoder must be a registered name, a class or factory, or a "
        f"mapping from leaf name to either, got {encoder!r}. Registered "
        f"names are ENCODERS.names(); a class is taken as it is and the "
        f"checkpoint then records it under no name (see UnnamedEncoder)."
    )


def resolve_encoder(encoder: Any, leaf_name: str, leaves, *, default: str):
    """The builder for ONE leaf's tower.

    `encoder` is a registered name, "auto" for the leaf's own kind, an
    `UnnamedEncoder`, or a mapping from leaf name to any of those. `default`
    is the name "auto" means for this leaf, which its spec's kind decides and
    this module does not read.
    """
    if isinstance(encoder, dict):
        if leaf_name not in encoder:
            raise ValueError(
                f"encoder names {sorted(encoder)} name no encoder for leaf "
                f"{leaf_name!r}; this observation's leaves are {list(leaves)}"
            )
        encoder = encoder[leaf_name]
    if isinstance(encoder, UnnamedEncoder):
        return encoder.resolve()
    return ENCODERS.resolve(default if encoder == "auto" else encoder)
