"""The Polyak blend that moves SAC's target critics.

Its own module, beside the losses rather than inside the agent, because it is
the one piece of SAC that is neither a loss nor bookkeeping: it is how the
lagged copies the target depends on are kept lagged. Two invariants live here
and both fail silently, which is why they are written down beside the code
that holds them rather than in the caller.
"""

from __future__ import annotations

from ttrl.engine.autograd import no_grad
from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.ops import add, mul
from ttrl.engine.tensor import copy_tensor_in_place
from ttrl.engine.trace import capture

__all__ = ["PolyakSync", "check_pairing"]


def check_pairing(nets) -> None:
    """Raise unless every target parameter pairs with its critic's by name.

    Pairs are matched BY NAME, not by position: a positional blend would
    cross-wire two different layers of equal shape, and the resulting target
    is a plausible-looking tensor that no shape check can catch.

    Separate from the blend because `PolyakSync` captures the blend's device
    work once and replays it. A check inside the captured region would run on
    the capture pass alone, so this runs on the host on every call, where the
    structures it is checking can still have changed.
    """
    for critic, target in zip(nets.critics, nets.targets, strict=True):
        pairs = zip(target.named_parameters(), critic.named_parameters(),
                    strict=True)
        for (dst_name, _dst), (src_name, _src) in pairs:
            if dst_name != src_name:
                raise ValueError(
                    f"SAC: critic and target structures differ "
                    f"({src_name!r} vs {dst_name!r}); a positional blend "
                    f"would cross-wire two different layers"
                )


def _blend(nets, tau: float) -> None:
    """Device work only. Address stability: the blend is computed into
    temporaries and copied into each target buffer, which is never rebound."""
    for critic, target in zip(nets.critics, nets.targets, strict=True):
        pairs = zip(target.parameters(), critic.parameters(), strict=True)
        with no_grad():
            for dst, src in pairs:
                copy_tensor_in_place(dst, add(mul(dst, 1.0 - tau), mul(src, tau)))


class PolyakSync:
    """target <- tau * critic + (1 - tau) * target, in place, eagerly or
    replayed from a captured trace.

    The blend is a fixed sequence of elementwise ops over buffers that are
    never rebound, and it reads nothing back to host, so it captures as it
    stands. The name pairing is the one part that must not be captured, and
    `check_pairing` runs on every call for that reason.
    """

    def __init__(self, nets, tau: float, device, *, traced: bool = False,
                 cfg: EngineConfig = DEFAULT):
        self.nets = nets
        self.tau = float(tau)
        self.device = device
        self.traced = bool(traced)
        self.cfg = cfg
        self._trace = None

    def run(self) -> None:
        check_pairing(self.nets)
        if not self.traced:
            _blend(self.nets, self.tau)
            return
        if self._trace is None:
            # Warm-up pass outside the capture, so capture records commands
            # rather than a first-time kernel compile. It is a real blend of
            # one step's worth, which is what this call owed anyway.
            _blend(self.nets, self.tau)
            self._trace, _ = capture(self.device,
                                     lambda: _blend(self.nets, self.tau))
            # Capture only records, so the first replay is what applies the
            # blend this call stands for. The warm-up above already applied
            # one, so the capture pass's own blend is skipped by replaying
            # exactly once here and not again until the next call.
            return
        self._trace.replay(cfg=self.cfg)

    def close(self) -> None:
        trace, self._trace = self._trace, None
        if trace is not None:
            trace.close()
