"""The Conv2d layer.

Geometry comes from the input shape handed to the constructor, so the layer
knows its own output shape before it is ever called and `__call__(x)` returns a
single Tensor. That is what lets a Sequential hold a convolution, and it is why
nothing here knows about 84x84 or four stacked frames.
"""

from __future__ import annotations

import math

import numpy as np
import ttnn

from ttrl.engine.autograd import Tensor
from ttrl.engine.conv.config import FP32
from ttrl.engine.conv.forward import build_plan, conv2d
from ttrl.engine.conv.geometry import ConvGeometry
from ttrl.engine.conv.packing import fold_weight_numpy, unfold_weight_numpy
from ttrl.engine.nn import Module
from ttrl.engine.ops.elementwise import add
from ttrl.engine.tensor import copy_in_place, restore_in_place, to_host

__all__ = ["Conv2d"]


class Conv2d(Module):
    """A convolution holding its own prepared master weight and optional bias.

    Parameters are discovered by `Module`, which walks attributes, so the
    weight and bias need no registration.

    `in_shape` is (channels, height, width), matching ObsSpec, and `batch` is
    fixed at construction because the shard placement is probed there, against
    the activation extent the layer will actually be handed.

    Init is PyTorch's nn.Conv2d default: uniform on [-1/sqrt(fan_in),
    +1/sqrt(fan_in)] with fan_in = in_channels * kernel * kernel, which is
    kaiming_uniform_(a=sqrt(5)) rather than textbook He-uniform.

    There is no `cfg`: nothing a convolution does depends on an EngineConfig
    any more. The three fields left on it are device-open and trace-replay
    facts, and the one that used to reach a conv (conv_act_block_h) is deleted.

    `profile`, the DeviceProfile of `device`, supplies the arch name the dX
    row ceiling is calibrated against. It is optional only so that a caller
    holding just a device still works; the arch then comes from the same ttnn
    call DeviceProfile makes.

    `needs_input_grad=False` skips the dX machinery entirely, which a first
    conv layer should pass: its input is the observation, a leaf, so dX is
    never computed, and dX is the part of the layer that stops fitting in L1
    first (conv1's does not from batch 64 up). A backward that then does reach
    for dX raises rather than silently returning no gradient.

    The weight is stored in ttnn.conv2d's prepared layout, which is the layout
    the convolution consumes, so the optimizer updates that tensor in place and
    nothing re-prepares it. `weight_numpy` and `load_weight_numpy` are the
    (host-side, never in a step) conversions to and from logical OIHW, and
    `state_dict`/`load_state_dict` below are overridden to use them, so a
    checkpoint carries the portable OIHW weight rather than the prepared
    matrix. Saving the prepared buffer verbatim would tie the checkpoint to a
    padding that is a function of the shard and parallel config and of the ttnn
    build: conv1's prepared buffer holds 16384 elements against 2048 logical
    ones.
    """

    def __init__(self, in_shape, out_channels: int, kernel: int, device, *, batch: int,
                 stride: int = 1, padding: int = 0, bias: bool = True, fold: int = 1,
                 input_scale: float = 1.0, weight=None,
                 needs_input_grad: bool = True, profile=None):
        if len(tuple(in_shape)) != 3:
            raise ValueError(
                f"Conv2d: in_shape must be (channels, height, width), got {tuple(in_shape)}"
            )
        in_channels, in_h, in_w = (int(v) for v in in_shape)
        self.geom = ConvGeometry(
            batch=int(batch), in_channels=in_channels, in_h=in_h, in_w=in_w,
            out_channels=int(out_channels), kernel=int(kernel), stride=int(stride),
            padding=int(padding), fold=int(fold),
        )
        self.device = device
        self.input_scale = float(input_scale)
        if weight is None:
            bound = 1.0 / math.sqrt(in_channels * kernel * kernel)
            weight = np.random.uniform(
                -bound, bound, (out_channels, in_channels, kernel, kernel)
            ).astype(np.float32)
        else:
            weight = np.asarray(weight, np.float32)
            want = (out_channels, in_channels, kernel, kernel)
            if weight.shape != want:
                raise ValueError(f"Conv2d: weight must have shape {want}, got {weight.shape}")
        self._oihw_shape = weight.shape
        master, self.plan = build_plan(
            self._to_dispatch_layout(weight), self.geom, device,
            input_scale=self.input_scale, needs_input_grad=needs_input_grad,
            profile=profile,
        )
        self.weight = Tensor(master, requires_grad=True)
        if bias:
            bound = 1.0 / math.sqrt(in_channels * kernel * kernel)
            b = np.random.uniform(-bound, bound, (1, 1, 1, out_channels)).astype(np.float32)
            self.bias = Tensor.from_numpy(b, device, requires_grad=True, dtype=FP32)
        else:
            self.bias = None

    # -- geometry ----------------------------------------------------------

    @property
    def out_shape(self) -> tuple[int, int, int]:
        return self.geom.out_shape

    @property
    def out_numel(self) -> int:
        return self.geom.batch * int(np.prod(self.geom.out_shape))

    def __repr__(self) -> str:
        return (f"Conv2d({self.geom.in_channels}x{self.geom.in_h}x{self.geom.in_w} -> "
                f"{self.out_shape}, k{self.geom.kernel}, s{self.geom.stride}, "
                f"p{self.geom.padding}, fold={self.geom.fold})")

    # -- parameters --------------------------------------------------------

    def _to_dispatch_layout(self, w_oihw: np.ndarray) -> np.ndarray:
        """Logical OIHW -> the OIHW of the convolution actually dispatched.

        A space-to-depth layer dispatches a (kernel/fold) convolution over
        fold*fold times as many input channels. The fold is a pure reindex of
        the same kernel elements, so it composes into the host-side packing and
        costs nothing per step.
        """
        if self.geom.fold == 1:
            return w_oihw
        return fold_weight_numpy(w_oihw, self.geom.fold)

    def weight_numpy(self) -> np.ndarray:
        """The logical (O, C, KH, KW) weight. Host read: never call in a step."""
        packed = to_host(self.weight.data).reshape(self.plan.pspec.layout.shape)
        w = self.plan.pspec.layout.unpack(packed)
        if self.geom.fold == 1:
            return w
        return unfold_weight_numpy(w, self.geom.fold, self.geom.in_channels,
                                   self.geom.kernel)

    def weight_grad_numpy(self) -> np.ndarray:
        """This step's dW in logical (O, C, KH, KW) form. Host read.

        dW lands in the prepared layout, the same one the master weight is
        stored in, so reshaping it to OIHW directly is wrong: for k=3 with
        C <= 16 ttnn pads each kernel row to a tile. It goes back through the
        same layout that packed the weight.
        """
        if self.weight.grad is None:
            raise ValueError("Conv2d.weight_grad_numpy: no gradient has been accumulated")
        packed = to_host(self.weight.grad).reshape(self.plan.pspec.layout.shape)
        w = self.plan.pspec.layout.unpack(packed)
        if self.geom.fold == 1:
            return w
        return unfold_weight_numpy(w, self.geom.fold, self.geom.in_channels,
                                   self.geom.kernel)

    def load_weight_numpy(self, w_oihw) -> None:
        """Write a logical (O, C, KH, KW) weight into the master, in place.

        In place, because captured traces replay against fixed device
        addresses: rebinding .data would leave every replay convolving with the
        buffer the trace recorded.
        """
        w = np.asarray(w_oihw, np.float32)
        if w.shape != self._oihw_shape:
            raise ValueError(
                f"Conv2d.load_weight_numpy: expected shape {self._oihw_shape}, got {w.shape}"
            )
        packed = self.plan.pspec.layout.pack(self._to_dispatch_layout(w))
        host = ttnn.Tensor(packed, FP32).to(ttnn.TILE_LAYOUT)
        copy_in_place(self.weight.data, host.to(self.device))

    # -- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """The portable form: `weight` in logical OIHW, `bias` as stored.

        Overridden together with `load_state_dict`, which is the protocol by
        which `Module` delegates this subtree whole instead of saving the
        prepared device matrix.
        """
        state = {"weight": self.weight_numpy()}
        if self.bias is not None:
            state["bias"] = self.bias.numpy()
        return state

    def load_state_dict(self, state: dict, device=None) -> None:
        """Restore from the portable form, in place.

        Validates the whole key set and every shape before writing anything,
        so a refused restore leaves the layer as it was rather than half
        written. `device` is accepted for the frozen signature and unused: the
        destination buffers already carry their device.
        """
        reference = self.state_dict()
        got, expected = set(state.keys()), set(reference)
        if got != expected:
            raise ValueError(
                "load_state_dict: checkpoint keys do not match module "
                f"parameters (missing={sorted(expected - got)}, "
                f"extra={sorted(got - expected)})"
            )
        for name, want in reference.items():
            shape = tuple(int(d) for d in np.asarray(state[name]).shape)
            if shape != want.shape:
                raise ValueError(
                    f"load_state_dict: shape mismatch for {name!r}: "
                    f"checkpoint has {shape}, parameter expects {want.shape}"
                )
        self.load_weight_numpy(state["weight"])
        if self.bias is not None:
            restore_in_place(self.bias.data, state["bias"])

    def at_batch(self, batch: int) -> "Conv2d":
        """This layer at another activation batch, sharing weight and bias.

        A convolution fixes its batch at construction, so a forward narrower
        than the one the layer was built for needs its own plan. The prepared
        weight layout is a function of the channel and kernel extents and of
        the shard placement, none of which is the batch, so both views
        convolve with one master weight and an optimizer step is visible
        through both. That equality is checked rather than assumed: a weight
        prepared under another layout computes silently wrong output.
        """
        g = self.geom
        view = Conv2d(
            (g.in_channels, g.in_h, g.in_w), g.out_channels, g.kernel, self.device,
            batch=int(batch), stride=g.stride, padding=g.padding,
            bias=self.bias is not None, fold=g.fold, input_scale=self.input_scale,
            weight=self.weight_numpy(), needs_input_grad=False,
        )
        mine, theirs = self.plan.pspec, view.plan.pspec
        if (mine.layout.shape, mine.layout.cp, mine.shard_layout) != (
                theirs.layout.shape, theirs.layout.cp, theirs.shard_layout):
            raise ValueError(
                f"Conv2d.at_batch({batch}): this layer prepares its weight as "
                f"{mine.layout.shape} cp={mine.layout.cp} on {mine.shard_layout} "
                f"at batch {g.batch}, but as {theirs.layout.shape} "
                f"cp={theirs.layout.cp} on {theirs.shard_layout} at batch "
                f"{batch}, so one master weight cannot serve both. Run this "
                f"forward at the batch the layer was built for."
            )
        ttnn.deallocate(view.weight.data)
        view.weight = self.weight
        view.bias = self.bias
        return view

    # -- call --------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        out = conv2d(x, self.weight, self.plan)
        if self.bias is None:
            return out
        # out.data is [1, 1, N*out_h*out_w, C_out] and bias is [1, 1, 1, C_out],
        # which broadcasts and reduces on the last axis exactly like Linear's.
        return add(out, self.bias)
