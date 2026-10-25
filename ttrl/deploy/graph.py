"""The same exported policy as an ONNX graph, written from the artifact.

WHY THE ARTIFACT AND NOT THE CHECKPOINT. The graph is built from the `.npz`
`ttrl.export_policy` wrote, so there is one weight-extraction path and one
action map per policy family, and the ONNX file provably computes the function
of the file the robot already has. Two readers of a checkpoint would be two
chances to disagree about which encoder is the actor's.

NO TORCH. The weights are plain numpy and the graph is built with `onnx`'s own
helpers, which is a protobuf library and nothing more. `onnx` is an optional
extra declared the way `ttrl.train.trackers` declares its sinks: the import
happens inside the call, the refusal names the extra that provides it, and
nothing here degrades to writing a file that is not ONNX.

THE NORMALISER IS EMITTED, NOT FOLDED. Folding a whitening transform into the
first weight is only valid while the transform is affine, and this one is not:
it clips the whitened observation at `+/-clip`, which is exactly the part that
fires when a deployed robot leaves the distribution the weights were fitted
on. A graph whose input were a pre-normalised observation would also be the
trap this format exists to avoid, because the deployer would have to
reimplement Welford statistics and the clip correctly, in their own language,
from no stated contract. So the graph's ONE input is the environment's RAW
observation and the transform is `Cast -> Sub -> Div -> Clip -> Cast` nodes
inside it. The two casts are not decoration: `DeployedPolicy.normalize`
whitens in float64 and clips before narrowing to float32, which is what the
training wrapper does, so the graph does it in that order too and the two
agree to the last bit before the first matmul.

A MASKED POLICY'S GRAPH STILL TAKES ONE INPUT. The legal-action mask is the
tail of the observation, so the graph slices it off `obs` and sends the
illegal logit columns to -inf through `Where` before the `ArgMax`, which is
what the numpy artifact does. Emitting the plain `ArgMax` for a masked policy
would be a graph that commands illegal actions and looks correct.

OPSET 18, at the oldest IR version that can carry it. Every op emitted here
(MatMul, Add, Relu, Sub, Div, Clip, Cast, Tanh, Mul, Min, Max, ArgMax, Slice,
Greater, Where, Conv, Transpose, Flatten, Identity) had its final definition at or before opset 16,
so 18 is not reached for a feature: it is chosen as the floor every current
consumer accepts. onnxruntime has supported it since 1.14, TensorRT 10 and
OpenVINO 2024 import it, and it is comfortably above opset 11, which is where
`Clip` moved its bounds from attributes to inputs. Pinning lower would mean
emitting the attribute form of `Clip`; pinning higher buys nothing and cuts
off consumers.
"""

from __future__ import annotations

import json

import numpy as np

import ttrl
from ttrl.deploy.artifact import ACTION_MAPS
from ttrl.deploy.conv import emit_conv
from ttrl.deploy.read import read_artifact

__all__ = ["OPSET", "REQUIRES", "export_onnx"]

#: The opset the graph declares. See the module docstring for why this one.
OPSET = 18

#: The package this writer imports and the extra that installs it. ONE
#: source, as in `ttrl.train.trackers.REQUIRES`: the refusal message is built
#: from it, so what is imported and what a user is told to install cannot
#: drift.
REQUIRES = ("onnx", "onnx")


def _onnx():
    """The `onnx` module, or the refusal naming the extra that provides it."""
    package, extra = REQUIRES
    try:
        # Optional dependency, imported on first use; the refusal below names the extra.
        import onnx
        from onnx import helper, numpy_helper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"ttrl.export_onnx needs the {package} package, which is not "
            f"installed ({exc.name}). Install it with: pip install "
            f"'tt-rl[{extra}]'. The numpy artifact ttrl.export_policy writes "
            f"needs no extra and ttrl.load_policy reads it."
        ) from exc
    return onnx, helper, numpy_helper


def export_onnx(artifact_path: str, out_path: str) -> str:
    """Write the policy in `artifact_path` as an ONNX graph, and return it.

    `artifact_path` is a file `ttrl.export_policy` wrote. The graph takes one
    input, `obs`, float32 of shape `(batch, *obs_shape)`: `(batch, obs_dim)`
    for a vector policy, `(batch, C, H, W)` in raw pixel units for an image
    one. It emits `action`, float32 of shape `(batch, action_dim)` for a
    continuous policy in the environment's own units, or int64 of shape
    `(batch,)` for a discrete one, holding the chosen action's index. A
    second output, `head`, is the head's raw row before the action map: the
    Q values or logits a discrete action was chosen from.
    """
    onnx, helper, numpy_helper = _onnx()
    meta, arrays = read_artifact(artifact_path)
    builder = _Builder(helper, numpy_helper, onnx.TensorProto)
    row = builder.policy(meta, arrays)
    output = builder.action(meta, arrays, row)
    builder.node("Identity", [row], "head")
    head = helper.make_tensor_value_info(
        "head", onnx.TensorProto.FLOAT, ["batch", int(arrays["head.bias"].shape[-1])])
    graph = helper.make_graph(
        builder.nodes, f"ttrl_{meta['algo']}_policy",
        [helper.make_tensor_value_info(
            "obs", onnx.TensorProto.FLOAT, ["batch", *meta["obs_shape"]])],
        [output, head], initializer=builder.initializers)
    opset = helper.make_operatorsetid("", OPSET)
    model = helper.make_model(graph, producer_name="ttrl",
                              opset_imports=[opset])
    # Pinned, not left at the installed library's newest. `make_model` stamps
    # whatever IR version this `onnx` release writes, and a consumer one
    # release behind refuses the file outright: onnxruntime 1.31 rejects the
    # IR 14 that onnx 1.23 stamps by default, whatever the opset says. The
    # oldest IR that can express opset 18 is what every consumer of opset 18
    # can read, and the library computes which that is.
    model.ir_version = helper.find_min_ir_version_for([opset])
    for key, value in _properties(meta, artifact_path).items():
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out_path)
    return out_path


def _properties(meta: dict, artifact_path: str) -> dict[str, str]:
    """What a consumer needs to know about the graph it was handed.

    The normaliser's description is included because a deployed graph that
    whitens inside itself gives its reader no other way to see that it does,
    and "is the transform in the graph" is the first question a sim-to-sim
    comparison has to answer.
    """

    return {
        "ttrl_version": str(ttrl.__version__),
        "ttrl_algo": str(meta["algo"]),
        "ttrl_action_map": str(meta["action_map"]),
        "ttrl_artifact_format": meta["artifact_format"],
        "ttrl_source_artifact": str(artifact_path),
        "ttrl_normalizer": json.dumps(meta["normalizer"]),
    }


class _Builder:
    """Nodes and initializers for one policy. Every emitted name is unique."""

    def __init__(self, helper, numpy_helper, types):
        self._helper = helper
        self._numpy_helper = numpy_helper
        self._types = types
        self.nodes: list = []
        self.initializers: list = []

    def constant(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(self._numpy_helper.from_array(
            np.ascontiguousarray(array), name))
        return name

    def node(self, op: str, inputs: list[str], out: str, **attrs) -> str:
        self.nodes.append(self._helper.make_node(op, inputs, [out], **attrs))
        return out

    # -- the observation transform ----------------------------------------

    def normalized(self, meta: dict, arrays: dict) -> str:
        """`obs` through the frozen transform, in float32, ready for the trunk."""
        if meta["normalizer"] is None:
            # Identity rather than wiring `obs` straight into the first
            # MatMul: one name downstream means the trunk is built the same
            # way whether a transform was applied or not.
            return self.node("Identity", ["obs"], "net_input")
        clip = float(meta["normalizer"]["clip"])
        wide = self.node("Cast", ["obs"], "obs_f64", to=self._types.DOUBLE)
        centred = self.node(
            "Sub", [wide, self.constant("obs_mean", arrays["obs_mean"])],
            "obs_centred")
        whitened = self.node(
            "Div", [centred, self.constant("obs_std", arrays["obs_std"])],
            "obs_whitened")
        clipped = self.node(
            "Clip",
            [whitened,
             self.constant("clip_lo", np.array(-clip, np.float64)),
             self.constant("clip_hi", np.array(clip, np.float64))],
            "obs_clipped")
        return self.node("Cast", [clipped], "net_input", to=self._types.FLOAT)

    # -- the network ------------------------------------------------------

    def policy(self, meta: dict, arrays: dict) -> str:
        """The head's raw row: the trunk's `Linear -> ReLU` stack, then the head.

        The weights are stored `(in_features, out_features)`, which is
        MatMul's own layout, so nothing here transposes anything.
        """
        hidden = self.normalized(meta, arrays)
        if meta["conv"]:
            hidden = emit_conv(self, meta, arrays, hidden)
        for index in range(int(meta["trunk_layers"])):
            hidden = self.linear(f"trunk_{index}", hidden, arrays,
                                 f"trunk.{index}")
            hidden = self.node("Relu", [hidden], f"trunk_{index}_relu")
        return self.linear("head", hidden, arrays, "head")

    def linear(self, name: str, source: str, arrays: dict, key: str) -> str:
        product = self.node(
            "MatMul", [source, self.constant(f"{name}_w", arrays[f"{key}.weight"])],
            f"{name}_matmul")
        return self.node(
            "Add", [product, self.constant(f"{name}_b", arrays[f"{key}.bias"])],
            f"{name}_out")

    # -- the action map ---------------------------------------------------

    def legal(self, meta: dict, row: str) -> str:
        """`row` with the illegal columns at -inf, or `row` for no mask.

        The mask is the tail of the graph's own `obs` input, so masking adds no
        input: it is a Slice of the observation the trunk already read, and the
        same -inf `DeployedPolicy.__call__` substitutes.
        """
        width = int(meta["action_mask"])
        if not width:
            return row
        mask = self.node(
            "Slice",
            ["obs",
             self.constant("mask_start", np.array([-width], np.int64)),
             self.constant("mask_end", np.array([np.iinfo(np.int64).max],
                                                np.int64)),
             self.constant("mask_axis", np.array([1], np.int64))],
            "legal_mask")
        legal = self.node(
            "Greater", [mask, self.constant("mask_half", np.array(0.5, np.float32))],
            "legal_bool")
        return self.node(
            "Where",
            [legal, row,
             self.constant("illegal", np.array(-np.inf, np.float32))],
            "masked_row")

    def action(self, meta: dict, arrays: dict, row: str):
        """The action, and the graph output describing it.

        One branch per entry in `ttrl.deploy.artifact.ACTION_MAPS`, computing
        the same arithmetic `DeployedPolicy.__call__` computes. An unknown map
        raises: a graph is not emitted for a function this writer cannot spell.
        """
        action_map = meta["action_map"]
        if action_map == "argmax":
            self.node("ArgMax", [self.legal(meta, row)], "action", axis=-1,
                      keepdims=0)
            return self._helper.make_tensor_value_info(
                "action", self._types.INT64, ["batch"])
        if action_map == "tanh_bounds":
            low, high = arrays["action_low"], arrays["action_high"]
            squashed = self.node("Tanh", [row], "squashed")
            scaled = self.node(
                "Mul", [squashed,
                        self.constant("half_width", (high - low) * np.float32(0.5))],
                "scaled")
            self.node(
                "Add", [scaled,
                        self.constant("centre", (high + low) * np.float32(0.5))],
                "action")
        elif action_map == "clip_bounds":
            # Min/Max rather than Clip: the bounds are per action dimension
            # and Clip's are scalars.
            floored = self.node(
                "Max", [row, self.constant("action_low", arrays["action_low"])],
                "floored")
            self.node(
                "Min", [floored,
                        self.constant("action_high", arrays["action_high"])],
                "action")
        else:
            raise ValueError(
                f"this writer emits no graph for action map {action_map!r}; it "
                f"knows {sorted(ACTION_MAPS)}"
            )
        return self._helper.make_tensor_value_info(
            "action", self._types.FLOAT, ["batch", int(meta["action_dim"])])
