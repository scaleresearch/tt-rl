"""The encoder of a whole observation: one tower per leaf, features concatenated.

`build_encoder` lives here and is the single reader of `ObsSpec.kind` in the
package: it resolves one encoder per leaf, by name or from that leaf's kind,
and wraps them in a `MultiEncoder`. A one-leaf observation gets one tower in a
`MultiEncoder` of one, so the common case and the general case are the same
module tree and the same code; nothing in the package builds a tower that is
not inside one.

CONCATENATION ORDER IS THE GROUP'S DECLARATION ORDER. The per-leaf features
are joined in `ObsGroup.names` order, not in the order a mapping happened to
be built, because the head that reads the joined row has one weight column per
feature and a permuted order trains on scrambled features with no error
anywhere. `ttrl.spaces.group.ObsGroup.observation` puts every observation into
the same order for the same reason.

`concat`'s backward slices the incoming gradient back to each operand's own
extent, so each leaf's tower receives exactly its own columns of the head's
gradient and nothing is dropped or shared. `ttrl.engine.ops.shape.split` is
the same cut in the forward direction.
"""

from __future__ import annotations

from ttrl.engine.nn import Module, ModuleDict
from ttrl.engine.ops import concat
from ttrl.nets.encoders import DEFAULT_ENCODER_BY_KIND
from ttrl.nets.fusion import DEFAULT_CRITIC_ENCODER_BY_KIND
from ttrl.nets.choice import resolve_encoder
from ttrl.spaces import require_group

__all__ = ["DEFAULT_ENCODER_BY_ROLE", "MultiEncoder", "build_encoder"]


#: What 'auto' means, per ROLE and then per observation kind. A tower's role
#: is a real input to the choice and not a wrapper's detail: a critic has an
#: action to fuse onto its features, so on a vector leaf its default is
#: 'identity', which puts `concat(s, a)` on the critic's input, while every
#: other tower defaults to the leaf's own architecture. An image leaf resolves
#: to its conv tower in BOTH roles, because an action has no spatial extent to
#: occupy in a plane of pixels and a pixel critic must fuse after the tower.
#:
#: Both halves are DEFAULTS and neither is a fallback: a name resolved here is
#: the name the leaf's declared kind implies, it is reached only when the
#: config says 'auto', and nothing recovers from a bad name by substituting
#: one. An unknown role raises rather than picking a table.
DEFAULT_ENCODER_BY_ROLE = {
    "observation": DEFAULT_ENCODER_BY_KIND,
    "critic": DEFAULT_CRITIC_ENCODER_BY_KIND,
}


class MultiEncoder(Module):
    """One encoder per leaf; `forward` concatenates their features.

    `out_features` is the sum of the towers' widths, so a head is sized from
    the joined row the same way it is sized from a single tower's.
    """

    #: The attribute the towers hang off, and so the segment leaf names add to
    #: a parameter's dotted name. Named here because the checkpoint reader of
    #: a pre-leaf file has to insert exactly this segment.
    SLOT = "towers"

    def __init__(self, towers: dict, names):
        self._names = tuple(names)
        if set(self._names) != set(towers) or not self._names:
            raise ValueError(
                f"MultiEncoder: towers {sorted(towers)} do not match the leaf "
                f"names {list(self._names)}. Give encoder= one entry per leaf "
                f"of the observation, keyed by leaf name."
            )
        # Declaration order, so the dotted parameter names and the
        # concatenation both follow the spec rather than a dict's history.
        setattr(self, self.SLOT, ModuleDict({n: towers[n] for n in self._names}))
        self.out_features = sum(int(towers[n].out_features) for n in self._names)

    @property
    def names(self) -> tuple[str, ...]:
        """Leaf names in concatenation order."""
        return self._names

    @property
    def towers_by_name(self) -> ModuleDict:
        return getattr(self, self.SLOT)

    def __getitem__(self, leaf: str):
        """One leaf's tower, by leaf name."""
        return self.towers_by_name[leaf]

    def __len__(self) -> int:
        return len(self._names)

    def forward(self, obs: dict):
        towers = self.towers_by_name
        missing = [n for n in self._names if n not in obs]
        if missing:
            raise KeyError(
                f"MultiEncoder was given leaves {sorted(obs)} and needs "
                f"{list(self._names)} (missing {missing})"
            )
        features = [towers[name](obs[name]) for name in self._names]
        # One leaf is one row already: concatenating a single operand would
        # add a tape node that copies it unchanged.
        return features[0] if len(features) == 1 else concat(features, dim=-1)


def build_encoder(obs_group, device, *, batch: int, name="auto",
                  role: str = "observation", **kwargs) -> MultiEncoder:
    """Resolve one encoder per leaf and join them. The one reader of `kind`.

    `name` is anything `ttrl.nets.choice.resolve_encoder` takes: "auto", a
    registered name, a class handed over directly, or a mapping from leaf
    name to any of those. Remaining keyword arguments reach every leaf's
    builder. The one `kind` read in the package is the default this passes
    for "auto".

    `role` picks which table of `DEFAULT_ENCODER_BY_ROLE` that default comes
    out of, and so is only consulted when `name` is "auto": a tower that was
    named is built under that name whatever its role. "critic" is the role
    with an action to fuse afterwards, which is what makes its default differ
    on a vector leaf. An unrecognised role raises here rather than quietly
    taking the observation table.

    No `cfg`: nothing an encoder builds reads an `EngineConfig`, whose three
    remaining fields are all device-open and trace-replay facts settled before
    any layer exists. A third-party builder that wants one still receives it
    through `**kwargs`, so dropping the dedicated parameter costs no
    capability.
    """
    group = require_group(obs_group, "build_encoder")
    if role not in DEFAULT_ENCODER_BY_ROLE:
        raise ValueError(
            f"build_encoder: role must be one of "
            f"{sorted(DEFAULT_ENCODER_BY_ROLE)}, got {role!r}. The role "
            f"decides what 'auto' means for a leaf, so there is no default "
            f"role to fall back to."
        )
    defaults = DEFAULT_ENCODER_BY_ROLE[role]
    towers = {}
    for leaf_name, spec in group.specs():
        builder = resolve_encoder(name, leaf_name, group.names,
                                  default=defaults[spec.kind])
        towers[leaf_name] = builder(spec, device, batch=batch, **kwargs)
    return MultiEncoder(towers, group.names)
