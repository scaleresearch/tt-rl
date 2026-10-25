"""What an agent is allowed to emit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionKind = Literal["discrete", "continuous", "multi_discrete"]

__all__ = ["ActionSpec", "ActionKind"]


@dataclass(frozen=True)
class ActionSpec:
    """What an agent is allowed to emit.

    Three kinds. A `discrete` action is one index out of `n`. A `continuous`
    action is a vector of `dim` bounded scalars. A `multi_discrete` action is
    a tuple of `len(nvec)` indices, the i-th out of `nvec[i]`, which is what a
    game whose action factorizes has: a poker bet is (action type, size
    bucket). A MultiBinary action is the multi_discrete whose every factor has
    size 2, and `ActionSpec.multi_binary` builds exactly that.

    Bounds are stored as tuples so the spec stays hashable and comparable,
    which matters because specs are checkpointed and compared on resume.
    `nvec` is defaulted to None and read back through the constructor, so a
    spec written before this field existed rebuilds equal to a discrete or
    continuous spec written after it, so adding it left every file on disk
    loadable.
    """

    kind: ActionKind
    n: int | None = None
    dim: int | None = None
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    nvec: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind == "discrete":
            if self.n is None or self.n < 1:
                raise ValueError(f"discrete actions need n >= 1, got {self.n}")
            self._forbid(("dim", "low", "high", "nvec"))
        elif self.kind == "continuous":
            if self.dim is None or self.dim < 1:
                raise ValueError(f"continuous actions need dim >= 1, got {self.dim}")
            self._forbid(("n", "nvec"))
            if self.low is None or self.high is None:
                raise ValueError("continuous actions need low and high bounds")
            if len(self.low) != self.dim or len(self.high) != self.dim:
                raise ValueError(
                    f"bounds must have length dim={self.dim}, "
                    f"got low={len(self.low)} high={len(self.high)}"
                )
            if any(hi <= lo for lo, hi in zip(self.low, self.high)):
                raise ValueError("every action bound must satisfy low < high")
        elif self.kind == "multi_discrete":
            if self.nvec is None:
                raise ValueError(
                    "multi_discrete actions need nvec, the size of each factor; "
                    "build one with ActionSpec.multi_discrete([n0, n1, ...])"
                )
            self._forbid(("n", "dim", "low", "high"))
            if len(self.nvec) < 2:
                raise ValueError(
                    f"multi_discrete actions need at least 2 factors, got "
                    f"nvec={self.nvec}. A one-factor composite action is a plain "
                    f"index: use ActionSpec.discrete({self.nvec[0] if self.nvec else 0})."
                )
            bad = [i for i, size in enumerate(self.nvec) if size < 2]
            if bad:
                raise ValueError(
                    f"every multi_discrete factor needs at least 2 values, got "
                    f"nvec={self.nvec} with factor(s) {bad} of size < 2. A factor "
                    f"with one value is a constant the agent cannot choose: drop it "
                    f"from nvec and let the environment supply it."
                )
        else:
            raise ValueError(
                f"action kind must be discrete, continuous or multi_discrete, "
                f"got {self.kind!r}"
            )

    def _forbid(self, names: tuple[str, ...]) -> None:
        set_ = [name for name in names if getattr(self, name) is not None]
        if set_:
            raise ValueError(f"{self.kind} actions must not carry {', '.join(set_)}")

    @property
    def width(self) -> int:
        """Number of scalars a policy head emits per action.

        For a factorized action that is the total over the factors: one head of
        width `sum(nvec)`, cut into one segment per factor. One head keeps the
        batch width fixed at trace capture and keeps the segment boundaries a
        build-time constant.
        """
        if self.kind == "discrete":
            return self.n
        if self.kind == "multi_discrete":
            return int(sum(self.nvec))
        return self.dim

    @property
    def factors(self) -> tuple[int, ...]:
        """The size of each factor of a composite action."""
        if self.nvec is None:
            raise ValueError(
                f"a {self.kind} ActionSpec has no factors; factors describe a "
                f"multi_discrete action"
            )
        return self.nvec

    @property
    def low_array(self) -> np.ndarray:
        if self.low is None:
            raise ValueError(f"{self.kind} action specs have no bounds")
        return np.asarray(self.low, dtype=np.float32)

    @property
    def high_array(self) -> np.ndarray:
        if self.high is None:
            raise ValueError(f"{self.kind} action specs have no bounds")
        return np.asarray(self.high, dtype=np.float32)

    @classmethod
    def discrete(cls, n: int) -> "ActionSpec":
        """One action out of `n`, named by its index."""
        return cls(kind="discrete", n=n)

    @classmethod
    def multi_discrete(cls, nvec) -> "ActionSpec":
        """A composite action: one index per factor, factor i out of nvec[i]."""
        sizes = np.asarray(nvec, dtype=np.int64).reshape(-1)
        return cls(kind="multi_discrete", nvec=tuple(int(v) for v in sizes))

    @classmethod
    def multi_binary(cls, n: int) -> "ActionSpec":
        """`n` independent on/off switches, as the multi_discrete it is."""
        if n < 2:
            raise ValueError(
                f"multi_binary needs n >= 2 switches, got {n}. One switch is "
                f"ActionSpec.discrete(2)."
            )
        return cls.multi_discrete((2,) * int(n))

    @classmethod
    def continuous(cls, low, high) -> "ActionSpec":
        """A vector of bounded scalars, one entry per `low`/`high` pair.

        The DIMENSION IS THE BOUNDS' OWN LENGTH and is not passed: a scalar
        low and high describe one entry, and `dim` bounded scalars are
        `continuous([-1.0] * dim, [1.0] * dim)`. Bounds are per entry because
        a joint's limits are its own.
        """
        lo = np.asarray(low, dtype=np.float32).reshape(-1)
        hi = np.asarray(high, dtype=np.float32).reshape(-1)
        if lo.shape != hi.shape:
            raise ValueError(f"low and high must have equal shape, got {lo.shape} and {hi.shape}")
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("continuous action bounds must be finite")
        return cls(
            kind="continuous",
            dim=int(lo.shape[0]),
            low=tuple(float(v) for v in lo),
            high=tuple(float(v) for v in hi),
        )
