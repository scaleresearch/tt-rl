"""Which chips a run may use, and how they are opened.

Its own module because it is its own concept: the run config says what to
train and for how long, this says what hardware that happens on. The
checkpoint format drops it for exactly that reason; see
`TrainConfig.UNSAVED_FIELDS`.

Imports nothing from `ttrl.engine`, so `ttrl.train.config` stays importable
with no ttnn present. `engine_config()` reaches for the engine lazily.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ttrl.engine.config import DEFAULT

__all__ = ["DeviceConfig"]


@dataclass
class DeviceConfig:
    """Which chips a run may use, and how they are opened.

    Chip ids here are PHYSICAL: the number in `/dev/tenstorrent/<n>`, which is
    what `ttrl doctor` prints. The local index into the visible set, which is
    what ttnn's own `device_id` means, is deliberately absent; see
    `ttrl.train.scoping`.
    """

    #: The physical chip this run scopes the process to, or None to take the
    #: lowest chip nobody else holds. A run must scope itself to one chip or
    #: it takes the whole board away from everyone else on it.
    chip: int | None = None
    l1_small_size: int = 32 * 1024
    trace_region_size: int = 32 * 1024 * 1024
    # "all" (every discovered chip, validated against a known mesh
    # descriptor, never inferred from a device count), an int chip count, or
    # an explicit list of physical chip ids. 1 is one chip opened by the
    # caller; anything else means the Trainer opens its own devices.
    chips: str | int | list[int] = 1
    # Learner updates between `ParamSync.average()` calls. Ignored at chips=1.
    sync_every: int = 50
    #: Seconds `ttrl.run` waits for a free chip before giving up. Mirrors
    #: `ttrl.engine.device.DEFAULT_WAIT_SECONDS`, which this layer must not
    #: import (`ttrl.train.config` stays ttnn-free); a test asserts the two
    #: are equal so neither can drift. The library never retries: a caller
    #: that wants a retry policy writes one around `run`.
    acquire_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.l1_small_size < 0:
            raise ValueError(f"l1_small_size must be >= 0, got {self.l1_small_size}")
        if self.trace_region_size < 0:
            raise ValueError(
                f"trace_region_size must be >= 0, got {self.trace_region_size}"
            )
        if self.sync_every < 1:
            raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
        if self.acquire_timeout <= 0:
            raise ValueError(
                f"acquire_timeout must be > 0 seconds, got {self.acquire_timeout}. "
                "It is how long a run waits for a free chip; raise it on a "
                "contended box, and run `ttrl doctor` to see who holds what."
            )
        if self.chip is not None and self.chip < 0:
            raise ValueError(f"chip must be >= 0 or None, got {self.chip}")
        if self.chip is not None and self.chips != 1:
            raise ValueError(
                f"device.chip={self.chip} scopes this process to one chip, but "
                f"device.chips={self.chips!r} asks for more than one. Set chip "
                "to None and name the set in chips, with TT_VISIBLE_DEVICES "
                "covering exactly that set."
            )
        # `chips` is resolved much later against real discovered hardware. Its
        # shape is knowable here, and a typo ("al", 0, []) must not survive
        # until a device is already being opened.
        if isinstance(self.chips, str):
            if self.chips != "all":
                raise ValueError(
                    f"chips must be 'all', an int, or a list of ints, got {self.chips!r}"
                )
        elif isinstance(self.chips, list):
            if not self.chips:
                raise ValueError("chips list must be non-empty")
            if any(
                not isinstance(c, int) or isinstance(c, bool) or c < 0
                for c in self.chips
            ):
                raise ValueError(
                    f"chips list must hold non-negative ints, got {self.chips!r}"
                )
            if len(set(self.chips)) != len(self.chips):
                raise ValueError(
                    f"chips list must not repeat a chip id, got {self.chips!r}"
                )
        elif isinstance(self.chips, int) and not isinstance(self.chips, bool):
            if self.chips < 1:
                raise ValueError(f"chips count must be >= 1, got {self.chips}")
        else:
            raise TypeError(
                f"chips must be 'all', an int, or a list of ints, got {self.chips!r}"
            )

    @property
    def replicas(self) -> int | None:
        """Replica count, or None when it is only knowable against hardware."""
        if isinstance(self.chips, int):
            return self.chips
        if isinstance(self.chips, list):
            return len(self.chips)
        return None

    def engine_config(self):
        """The `EngineConfig` these device settings describe.

        The single conversion between the two configs that share
        `l1_small_size` and `trace_region_size`. Every caller that opens a
        device for a run goes through here instead of hand-copying the pair.
        """

        return dataclasses.replace(
            DEFAULT,
            l1_small_size=self.l1_small_size,
            trace_region_size=self.trace_region_size,
        )
