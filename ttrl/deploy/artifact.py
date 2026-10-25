"""The exported policy artifact: one `.npz`, read with numpy and nothing else.

Importing this module imports `numpy` and nothing else. Not ttnn, not a chip, not
`ttrl.engine`, not an environment suite: a robot's host runs the controller,
and a controller that needs the training stack installed to produce a torque
is not deployable. `tests/deploy/test_deployed_policy.py` holds that
line by importing it with every one of those blocked.

The file holds the whole function from a raw observation to an action in the
environment's own units. The observation is the ACTOR's: a field an
`ObsSpec` marks critic-only is privileged simulator state, and it is absent
from both the input width and the stored statistics.

  1. the frozen observation transform (a running normaliser's mean and
     standard deviation, its clip), or none when the trained stack applied
     none,
  2. for an image policy, its conv tower (`ttrl.deploy.conv`),
  3. the actor trunk's `Linear -> ReLU` stack, with `ObsSpec.scale` already
     folded into the first weight of the whole network, exactly as
     `ScaledLinear` and `Conv2d` fold it,
  4. one linear head, whose row is a Gaussian mean for a continuous policy
     and one logit per action for a discrete one,
  5. the ACTION MAP that turns that row into an action. Three of them, one
     per policy family, named in the file rather than inferred:

       tanh_bounds   `centre + half_width * tanh(row)`, the deterministic
                     action of SAC's squashed Gaussian.
       clip_bounds   `clip(row, low, high)`, the deterministic action of
                     PPO's unsquashed Gaussian, which clips its sample to
                     the bounds instead of squashing it.
       argmax        `argmax(row)`, the greedy action of a categorical PPO
                     policy or of DQN's Q head. Discrete actions have no
                     bounds, so no bounds are stored.

A LEGAL-ACTION MASK IS PART OF THE OBSERVATION, so a masked policy is still a
function of the observation vector alone. `ObsSpec.action_mask` declares that
the final n entries of the observation are a {0, 1} mask, and the actor reads
those entries like any other feature: the artifact stores the width, slices
the mask off the raw observation and drops the illegal columns to negative
infinity before the argmax, which is the same choice the device agent's
`epsilon_greedy_masked` makes over its own finite penalty. Masking applies to
the `argmax` map only, which is the only map over a finite action set.

The transform is FROZEN. Training updates the normaliser's statistics on
every observation; a deployed controller must not, or two robots started from
the same artifact compute different torques after a minute of standing still.

The weights are stored in the library's own `(in_features, out_features)`
layout, so `forward` is a plain `x @ W` here as it is on the device.
"""

from __future__ import annotations

import numpy as np

from ttrl.deploy.conv import conv_features, conv_stages

__all__ = ["ACTION_MAPS", "DeployedPolicy"]

#: Every action map this reader implements, and whether it needs bounds. ONE
#: source: the exporter picks a name from here, `__call__` dispatches on the
#: same names, and a test enumerates them rather than being handed a list.
ACTION_MAPS = {
    "tanh_bounds": True,
    "clip_bounds": True,
    "argmax": False,
}


class DeployedPolicy:
    """A policy's deterministic action, in numpy, for an exported artifact.

    `policy(obs)` takes one observation of shape `obs_shape` or a batch of
    shape `(batch, *obs_shape)`: `(obs_dim,)` for a vector policy, `(C, H, W)`
    as the environment emits it for an image one. The observation is the environment's RAW
    observation: the transform the weights were trained under is inside this
    object. A continuous policy returns an action of matching rank in the
    environment's own units; a discrete one returns the chosen action's index
    as an `int64`, a scalar for one observation and shape `(batch,)` for a
    batch, which is what a discrete environment's `step` takes.

    The spec-shaped attributes mirror `ttrl.spaces.ActionSpec` field for
    field, so the half that does not apply reads the same way on both:
    `action_dim` and `n_actions` are None for the other kind, and
    `action_low` and `action_high` raise for a discrete policy exactly as
    `ActionSpec.low_array` does. A discrete action has no bounds, and a
    deployed file inventing +/-1 for it would be read as a scale.

    `action_mask` is the width of the legal-action mask the observation
    carries, 0 for a policy over an unrestricted action set. When it is set,
    the final `action_mask` entries of every observation handed to this object
    must be the {0, 1} mask, and the returned action is always a legal one.
    """

    def __init__(self, meta: dict, arrays: dict[str, np.ndarray]):
        self.algo = meta["algo"]
        self.action_map = meta["action_map"]
        if self.action_map not in ACTION_MAPS:
            raise ValueError(
                f"this artifact's action map is {self.action_map!r}; this code "
                f"implements {sorted(ACTION_MAPS)}. The map is how the head's "
                f"row becomes an action, so there is no action to compute "
                f"without it."
            )
        self.action_kind = meta["action_kind"]
        self.action_mask = int(meta["action_mask"])
        if self.action_mask and self.action_map != "argmax":
            raise ValueError(
                f"this artifact declares a legal-action mask of width "
                f"{self.action_mask} and the action map {self.action_map!r}, "
                f"which is not over a finite action set, so there are no "
                f"columns the mask names. Export it again from its checkpoint: "
                f"ttrl.export_policy refuses that combination."
            )
        self.obs_dim = _or_none(meta["obs_dim"])
        self.obs_shape = tuple(int(d) for d in meta["obs_shape"])
        self._conv = conv_stages(meta, arrays)
        self.action_dim = _or_none(meta["action_dim"])
        self.n_actions = _or_none(meta["n_actions"])
        self._bounded = ACTION_MAPS[self.action_map]
        if self._bounded:
            self._low = arrays["action_low"]
            self._high = arrays["action_high"]
            self._centre = ((self._high + self._low) * np.float32(0.5))
            self._half_width = ((self._high - self._low) * np.float32(0.5))
        self._trunk = [
            (arrays[f"trunk.{i}.weight"], arrays[f"trunk.{i}.bias"])
            for i in range(int(meta["trunk_layers"]))
        ]
        self._head = (arrays["head.weight"], arrays["head.bias"])
        self.normalizer = meta["normalizer"]
        if self.normalizer is None:
            self._obs_mean = self._obs_std = self._clip = None
        else:
            self._obs_mean = arrays["obs_mean"]
            self._obs_std = arrays["obs_std"]
            self._clip = float(self.normalizer["clip"])

    # -- the action bounds, for the kind that has them --------------------

    @property
    def action_low(self) -> np.ndarray:
        return self._bound("action_low", 0)

    @property
    def action_high(self) -> np.ndarray:
        return self._bound("action_high", 1)

    def _bound(self, name: str, which: int) -> np.ndarray:
        if not self._bounded:
            raise ValueError(
                f"this is a {self.action_kind} policy over {self.n_actions} "
                f"actions and has no {name}: a choice among named actions has "
                f"no scale to be bounded on. Read n_actions."
            )
        return (self._low, self._high)[which]

    def _rows(self, obs) -> tuple[np.ndarray, bool]:
        array = np.asarray(obs)
        rank = len(self.obs_shape)
        single = array.ndim == rank
        if array.ndim not in (rank, rank + 1) or array.shape[-rank:] != self.obs_shape:
            raise ValueError(
                f"this policy takes an observation of shape {self.obs_shape} "
                f"or (batch, *{self.obs_shape}), got {array.shape}"
            )
        return array.reshape(-1, *self.obs_shape), single

    def normalize(self, obs) -> np.ndarray:
        """The network's input for `obs`: the frozen transform, or a copy.

        Public because a controller that logs one thing about its policy
        should log this: an input outside the clip for a sustained stretch is
        a robot operating off the distribution the weights were fitted on,
        and nothing downstream of here can tell.

        The arithmetic is float64 up to the clip and float32 after it, which
        is what the training wrapper does, so the two agree bit for bit.
        """
        rows, single = self._rows(obs)
        if self._obs_mean is None:
            out = rows.astype(np.float32)
        else:
            whitened = (rows.astype(np.float64) - self._obs_mean) / self._obs_std
            out = np.clip(whitened, -self._clip, self._clip).astype(np.float32)
        return out[0] if single else out

    def head(self, obs) -> np.ndarray:
        """The head's raw row for `obs`: a Gaussian mean, or one logit each.

        Public for the same reason `normalize` is: a saturated mean and a
        flat logit row are both things a deployed controller should be able
        to report, and neither survives the action map.
        """
        rows, single = self._rows(obs)
        hidden = self.normalize(rows)
        if self._conv:
            hidden = conv_features(hidden, self._conv)
        for weight, bias in self._trunk:
            hidden = np.maximum(hidden @ weight + bias, np.float32(0.0))
        weight, bias = self._head
        out = hidden @ weight + bias
        return out[0] if single else out

    def legal_mask(self, obs) -> np.ndarray:
        """The {0, 1} legal-action rows this observation carries.

        Public because a controller whose action set changes every step should
        be able to log which actions it was offered beside the one it chose.
        """
        if not self.action_mask:
            raise ValueError(
                f"this policy declares no legal-action mask, so the tail of its "
                f"observation is features rather than indicators and reading it "
                f"as a mask would be a guess. Every action it returns is legal. "
                f"Read action_mask to test for one."
            )
        rows, single = self._rows(obs)
        mask = np.ascontiguousarray(rows[:, -self.action_mask:], dtype=np.float32)
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError(
                f"the final {self.action_mask} entries of an observation handed "
                f"to this policy are its legal-action mask and must each be 0.0 "
                f"or 1.0, got values {np.unique(mask).tolist()[:8]}. Write the "
                f"mask there as indicators and keep every other feature ahead "
                f"of it."
            )
        empty = np.flatnonzero(~(mask > 0.5).any(axis=-1))
        if empty.size:
            raise ValueError(
                f"the observation in row {int(empty[0])} declares every action "
                f"illegal, so there is nothing to choose. Leave at least one "
                f"action legal in any state this policy is asked to act on, and "
                f"end the episode instead of presenting a state with no move."
            )
        return mask[0] if single else mask

    def __call__(self, obs):
        rows, single = self._rows(obs)
        row = self.head(rows)
        if self.action_map == "argmax":
            if self.action_mask:
                legal = self.legal_mask(rows) > 0.5
                row = np.where(legal, row, np.float32(-np.inf))
            action = np.argmax(row, axis=-1).astype(np.int64)
            return action[0] if single else action
        if self.action_map == "tanh_bounds":
            action = self._centre + self._half_width * np.tanh(row)
        else:
            action = np.clip(row, self._low, self._high)
        action = action.astype(np.float32)
        return action[0] if single else action

    def __getattr__(self, name: str):
        """The names a reader tries before reading, answered with the real one.

        A policy object with no `act` is a dead end for anyone who met one of
        this library's agents first, and the artifact is deliberately not an
        agent: there is one function here and calling it is the whole API.
        """
        if name in ("act", "action", "predict", "forward", "step"):
            raise AttributeError(
                f"DeployedPolicy has no {name!r}: call the policy directly, "
                f"policy(obs), which returns the deterministic action for one "
                f"observation or a batch of them. policy.normalize(obs) and "
                f"policy.head(obs) are the two steps inside that call."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def __repr__(self) -> str:
        transform = "none" if self.normalizer is None else "normalised"
        width = (f"action_dim={self.action_dim}" if self._bounded
                 else f"n_actions={self.n_actions}")
        return (
            f"DeployedPolicy(algo={self.algo!r}, obs_shape={self.obs_shape}, "
            f"{width}, action_map={self.action_map!r}, "
            f"trunk={len(self._trunk)}, transform={transform})"
        )


def _or_none(value):
    return None if value is None else int(value)
