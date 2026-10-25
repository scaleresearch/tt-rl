"""state_dict / load_state_dict for Adam, restoring moment buffers in place."""

from __future__ import annotations

import numpy as np

from ttrl.engine.tensor import restore_in_place, to_host

__all__ = ["AdamCheckpointMixin"]


class AdamCheckpointMixin:
    """Carries no state of its own; reads Adam's attributes directly."""

    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "steps": list(self.steps),
            "lr": self.lr,
            "betas": (self.b1, self.b2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            # Canonical Adam m and v, converted out of the stored S and V, so
            # a checkpoint never depends on how the update is composed.
            "m": [to_host(m) * self._m_scale for m in self.m],
            "v": [to_host(v) * self._v_scale for v in self.v],
        }

    def load_state_dict(self, state: dict, device=None) -> None:
        """Restore step counts and moment buffers in place.

        `device` is accepted for symmetry with `Module.load_state_dict` and is
        not needed: m and v are existing device buffers and are written into
        directly, so their addresses survive a restore.

        Everything is validated before any device buffer is touched, so a
        mismatched checkpoint raises instead of partially loading.
        """
        if len(state["m"]) != len(self.m) or len(state["v"]) != len(self.v):
            raise ValueError(
                f"Adam.load_state_dict: checkpoint has {len(state['m'])} m and "
                f"{len(state['v'])} v buffers, expected {len(self.m)} of each for "
                "this optimizer's parameter list."
            )
        for key, buffers in (("m", self.m), ("v", self.v)):
            for i, (arr, buf) in enumerate(zip(state[key], buffers)):
                got = tuple(np.asarray(arr).shape)
                want = tuple(int(d) for d in buf.shape)
                if got != want:
                    raise ValueError(
                        f"Adam.load_state_dict: {key}[{i}] has shape {got}, which does "
                        f"not match its parameter's shape {want}."
                    )

        t = int(state["t"])
        steps = [int(s) for s in state.get("steps", [t] * len(self.params))]
        lr = state.get("lr", self.lr)
        b1, b2 = state.get("betas", (self.b1, self.b2))
        eps = state.get("eps", self.eps)
        weight_decay = state.get("weight_decay", self.weight_decay)

        # Restored hyperparameters bypass __init__, so they get the same
        # validation. Without it a checkpoint carrying b2 == 1.0 divides by
        # zero inside the bias correction one step later, far from the cause.
        if len(steps) != len(self.params):
            raise ValueError(
                f"Adam.load_state_dict: checkpoint has {len(steps)} step counters, "
                f"expected {len(self.params)}."
            )
        if t < 0 or any(s < 0 for s in steps):
            raise ValueError(
                f"Adam.load_state_dict: step counts must be >= 0, got t={t}, steps={steps}."
            )
        if not lr > 0:
            raise ValueError(f"Adam.load_state_dict: lr must be > 0, got {lr}.")
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"Adam.load_state_dict: betas must be in [0, 1), got {(b1, b2)}.")
        if not eps > 0:
            raise ValueError(f"Adam.load_state_dict: eps must be > 0, got {eps}.")
        if weight_decay < 0:
            raise ValueError(
                f"Adam.load_state_dict: weight_decay must be >= 0, got {weight_decay}."
            )
        if (weight_decay != 0.0) != bool(self._wd_scratch):
            raise ValueError(
                f"Adam.load_state_dict: checkpoint has weight_decay={weight_decay} but "
                f"this optimizer was built with weight_decay={self.weight_decay}. The "
                "weight-decay scratch buffer is allocated at construction from that "
                "value, so it cannot be changed by a restore. Rebuild the optimizer "
                "with the checkpoint's weight_decay."
            )

        self.t = t
        self.steps = steps
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.eps = eps
        self.weight_decay = weight_decay
        # betas may have just changed, so the scales are read after assigning them.
        ms, vs = self._m_scale, self._v_scale
        for i, arr in enumerate(state["m"]):
            restore_in_place(self.m[i], np.asarray(arr) / ms)
        for i, arr in enumerate(state["v"]):
            restore_in_place(self.v[i], np.asarray(arr) / vs)
