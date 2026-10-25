"""TrainStep: one training step, run eagerly or replayed from a trace."""

from __future__ import annotations

from typing import Callable

import ttnn

from ttrl.engine.config import DEFAULT, EngineConfig
from ttrl.engine.step.inputs import InputBinder
from ttrl.engine.step.metrics import read_metrics
from ttrl.engine.trace import capture

__all__ = ["MODES", "StepFn", "TrainStep"]

StepFn = Callable[[dict], dict]

MODES = ("eager", "traced_fwd_bwd")


class TrainStep:
    """Wraps a step function behind one call API in either mode.

    `step_fn(inputs) -> dict[str, Tensor]` runs the forward pass, the loss and
    `backward()`, leaving gradients in each parameter's `.grad`. It must not
    call the optimizer, which TrainStep owns and always runs eagerly outside
    any traced region, and in `traced_fwd_bwd` it must not read any tensor to
    host, because trace capture forbids host reads. The metrics it returns are
    read back after the traced region, never during it.

    `inputs` holds the fixed Tensors the step function reads. Address
    stability: their identity, and so their device addresses, never change for
    this TrainStep's lifetime. Update their contents with `load_input`.

    Only forward and backward are captured. The optimizer stays outside: its
    bias-correction scalars are functions of the update count, which a
    captured trace would freeze at capture time and then reapply forever.
    Putting it inside the trace was measured at +1.3% to +1.7% against a 1.0%
    noise floor anyway, because those dispatches were never on the critical
    path; the host issues them while the device is still finishing the
    replayed backward.
    """

    def __init__(
        self,
        step_fn: StepFn,
        inputs: dict,
        device,
        *,
        optimizer=None,
        mode: str = "eager",
        cfg: EngineConfig = DEFAULT,
    ):
        if mode not in MODES:
            raise ValueError(f"TrainStep mode must be one of {list(MODES)}, got {mode!r}")
        self.step_fn = step_fn
        self.inputs = inputs
        self.device = device
        self.optimizer = optimizer
        self.mode = mode
        self.cfg = cfg
        self._binder = InputBinder(device, inputs)
        self._trace = None
        self._trace_metrics: dict | None = None
        self._captured_grads: list = []

    # -- inputs ----------------------------------------------------------

    def load_input(self, name: str, array) -> None:
        self._binder.load(name, array)

    # -- run -------------------------------------------------------------

    def run(self) -> dict:
        """Run one training step and return its metrics as host values."""
        if self.mode == "eager":
            return self._run_eager()
        return self._run_traced()

    def _run_eager(self) -> dict:
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        metrics = self.step_fn(self.inputs)
        if self.optimizer is not None:
            self.optimizer.step()
        return read_metrics(metrics)

    def _run_traced(self) -> dict:
        if self._trace is None:
            self._capture()
        else:
            self._trace.replay(cfg=self.cfg)
        self._rebind_grads()
        if self.optimizer is not None:
            self.optimizer.step()
        self._sync_after_replay()
        return read_metrics(self._trace_metrics)

    def _capture(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        # Warm-up pass: real dispatch, not part of the trace, so that capture
        # records commands rather than a first-time kernel compile. Its
        # gradients are discarded, since only the capture pass's backward
        # belongs in the trace.
        self.step_fn(self.inputs)
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        self._trace, self._trace_metrics = capture(
            self.device, lambda: self.step_fn(self.inputs)
        )
        if self.optimizer is not None:
            self._captured_grads = [(p, p.grad) for p in self.optimizer.params]
        # Capture only records. Nothing has executed, so the metric buffers
        # still hold whatever was in them at allocation time; run the trace
        # once, blocking, so this first call returns real numbers.
        self._trace.replay(blocking=True)

    def _rebind_grads(self) -> None:
        """Point every optimized parameter back at the gradient buffer this
        step's trace writes into.

        A replay re-executes recorded commands and runs no Python, so it
        writes the buffers `backward()` allocated at capture time while
        `param.grad` stays whatever Python last bound. With two TrainSteps
        over overlapping parameters, the second one's capture rebinds the
        first one's gradients, and the first optimizer would then step on the
        other region's gradient forever. Reinstating is an attribute
        assignment and costs no dispatch.
        """
        for param, grad in self._captured_grads:
            param.grad = grad

    def _sync_after_replay(self) -> None:
        """A non-blocking replay is ordered ahead of the metric readback on the
        same queue, so reading any metric is already the barrier. With no
        metrics there is nothing to order against, and the next iteration's
        `load_input` would overwrite buffers the replay is still reading."""
        if not self.cfg.trace_blocking and not self._trace_metrics:
            ttnn.synchronize_device(self.device)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release the captured trace and the staging buffers. Idempotent."""
        trace, self._trace = self._trace, None
        self._trace_metrics = None
        self._captured_grads = []
        try:
            if trace is not None:
                trace.close()
        finally:
            self._binder.close()

    def __enter__(self) -> "TrainStep":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
