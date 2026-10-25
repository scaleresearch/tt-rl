"""Observation and action descriptions.

These dataclasses are the seam that keeps the framework general. Every
component that would otherwise bake in a fact about one benchmark (encoder
choice, replay element dtype, head width, exploration strategy, evaluation
protocol) takes a spec instead. No component branches on which environment
suite produced the spec.

An observation is an `ObsGroup`: named leaves, each one an `ObsSpec` describing
a single array with its own shape, dtype and kind. One leaf is the
single-array case and is the default; two leaves are a camera frame plus
privileged simulator state. `ObsSpec` carries the division of one leaf between
the actor and the critic, and `ObsLeaf.actor` divides whole leaves the same
way, which together are what makes an asymmetric actor-critic expressible. See
`ttrl.spaces.group` and `ttrl.spaces.obs`.
"""

from __future__ import annotations

from ttrl.spaces.action import ActionKind, ActionSpec
from ttrl.spaces.group import DEFAULT_LEAF, ObsGroup, ObsLeaf, require_group
from ttrl.spaces.mask import check_legal, draw_legal, group_mask_rows, mask_rows
from ttrl.spaces.obs import ObsField, ObsKind, ObsSpec
from ttrl.spaces.observation import Observation

__all__ = ["ObsSpec", "ObsField", "ObsGroup", "ObsLeaf", "Observation",
           "DEFAULT_LEAF", "ActionSpec", "ObsKind", "ActionKind",
           "check_legal", "draw_legal", "group_mask_rows", "mask_rows",
           "require_group"]
