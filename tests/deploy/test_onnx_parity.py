"""The ONNX graph against the policy it claims to be, and nine ways to break it.

AN ONNX FILE THAT LOADS IS NOT A CORRECT POLICY. Every graph here is executed
under onnxruntime and its action compared against the numpy artifact's on the
same observations, half of them 40x outside the training distribution so the
normaliser's clip and tanh's saturation are both live. The one number each
comparison reports, the tolerance it is held to and the reasoning for that
tolerance are in `tests/deploy/_onnx.py`.

THE INJECTIONS ARE THE POINT. A parity test passes just as happily against a
weak comparison as against a sound graph, so each defect below is introduced
into a graph that was green a moment earlier and the same comparison must go
red. They are the defects an ONNX exporter actually ships: a weight in the
other layout, the normaliser missing or applied twice, a squash or a rescale
dropped, an argmax over the batch instead of the actions, an opset that
predates the op the graph uses.
"""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper
import pytest

from tests.deploy import _checkpoints as fixture
from tests.deploy import _onnx
from tests.deploy import _sac_checkpoint as sac_fixture
from tests.deploy.test_export_algos import COVERED, NORMALIZER
import ttrl
from ttrl.deploy.graph import OPSET
from ttrl.spaces import ActionSpec, ObsSpec

#: The square hidden widths every graph here is built with. Square so the
#: transposed-weight injection measures the arithmetic instead of being caught
#: on shape by the runtime before anything is computed.
HIDDEN = (8, 8)

CASES = {
    "sac": ("continuous", lambda: sac_fixture.agent_state()),
    "ppo_continuous": ("continuous",
                       lambda: fixture.ppo_state(fixture.CONTINUOUS, hidden=HIDDEN)),
    "ppo_discrete": ("discrete",
                     lambda: fixture.ppo_state(fixture.DISCRETE, hidden=HIDDEN)),
    "dqn": ("discrete", lambda: fixture.dqn_state(fixture.DISCRETE, hidden=HIDDEN)),
    "dqn_image": ("discrete", lambda: fixture.dqn_image_state()),
    "dqn_masked": ("discrete",
                   lambda: fixture.dqn_state(MASKED_ACTION_SPEC, hidden=HIDDEN,
                                             obs_dim=MASKED_OBS_DIM)),
}

#: The masked case's shapes. The observation must be wider than its mask,
#: since the mask is the tail of it, and it carries no normaliser: whitening
#: would rewrite the indicators the graph reads back.
MASKED_ACTIONS, MASKED_OBS_DIM = 3, 8
MASKED_ACTION_SPEC = ActionSpec.discrete(MASKED_ACTIONS)
MASKED_OBS_SPEC = ObsSpec.vector(MASKED_OBS_DIM, action_mask=MASKED_ACTIONS)


def masked_observations(rows: int = 256, seed: int = 5) -> np.ndarray:
    """Features beside every legal set with at least one action in it.

    Every legal set, because a batch whose masks never forbid the unmasked
    argmax is one an unmasked graph passes.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, MASKED_OBS_DIM - MASKED_ACTIONS)) * 2.0
    sets = np.array([m for m in np.ndindex(*([2] * MASKED_ACTIONS)) if any(m)],
                    np.float32)
    masks = sets[rng.integers(0, len(sets), rows)]
    return np.concatenate([features, masks], axis=1).astype(np.float32)


def build(tmp_path, case: str):
    """`(policy, ModelProto, observations)` for one case, all three green."""

    onnx_module, _runtime = _onnx.require_onnx()
    kind, state = CASES[case]
    tmp_path.mkdir(parents=True, exist_ok=True)
    masked = case.endswith("_masked")
    image = case.endswith("_image")
    spec = {"continuous": fixture.CONTINUOUS, "discrete": fixture.DISCRETE}[kind]
    obs_spec = (MASKED_OBS_SPEC if masked else
                fixture.IMAGE_SPEC if image else fixture.OBS_SPEC)
    source = sac_fixture.write(
        tmp_path / "ckpt.npz",
        env_policy_state=[] if masked or image else NORMALIZER, state=state(),
        obs_spec=obs_spec, action_spec=MASKED_ACTION_SPEC if masked else spec)
    artifact = ttrl.export_policy(source, str(tmp_path / "policy.npz"))
    graph = ttrl.export_onnx(artifact, str(tmp_path / "policy.onnx"))
    if image:
        obs = np.random.default_rng(11).integers(
            0, 256, (256, *obs_spec.shape)).astype(np.float32)
    else:
        obs = (masked_observations() if masked
               else _onnx.observations(fixture.OBS_DIM))
    return ttrl.load_policy(artifact), onnx_module.load(graph), obs


# -- parity -----------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_the_graph_computes_the_policy(tmp_path, case):
    policy, model, obs = build(tmp_path, case)
    if CASES[case][0] == "discrete":
        assert _onnx.min_decision_gap(policy, obs) > 1e-3, "an index match would be luck"
    assert _onnx.parity(policy, model, obs) <= _onnx.TOLERANCE
    if policy.action_mask:
        chosen = _onnx.run(model, obs)
        legal = policy.legal_mask(obs)[np.arange(len(obs)), chosen]
        assert (legal > 0.5).all(), "the graph commanded an illegal action"
        assert (chosen != np.argmax(policy.head(obs), axis=-1)).sum() >= 10, (
            "no observation in this batch forbids the unmasked argmax")


def test_every_covered_policy_has_a_parity_case():
    """The cases here cover what `ttrl.deploy.policies` claims, derived from
    the table rather than from this file's own keys."""
    assert {(fixture_algo(c), CASES[c][0]) for c in CASES} == set(COVERED)


def fixture_algo(case: str) -> str:
    return case.split("_")[0]


# -- the injections ---------------------------------------------------------

def bypass(model, outputs: set[str]):
    """Remove the nodes producing `outputs`, wiring each input straight through.

    The rewiring is transitive, so a whole chain can be cut at once, and if a
    removed node produced the graph's own output the node that now ends the
    graph is renamed onto it. That keeps every injection a VALID graph that
    computes the wrong thing, which is the kind that gets deployed.
    """
    rename: dict[str, str] = {}
    keep = []
    for node in model.graph.node:
        if node.output[0] in outputs:
            rename[node.output[0]] = node.input[0]
        else:
            keep.append(node)

    def resolve(name: str) -> str:
        while name in rename:
            name = rename[name]
        return name

    for node in keep:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
    if "action" in outputs:
        final = resolve("action")
        for node in keep:
            if node.output[0] == final:
                node.output[0] = "action"
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def initializer(model, name: str):
    return next(i for i in model.graph.initializer if i.name == name)


def replace_initializer(model, name: str, array: np.ndarray):
    target = initializer(model, name)
    target.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(array), name))
    return model


def transpose_a_weight(model):
    weight = numpy_helper.to_array(initializer(model, "trunk_1_w"))
    assert weight.shape[0] == weight.shape[1], (
        "a transpose of a rectangular weight is caught on shape by the "
        "runtime before any arithmetic happens, which proves nothing about "
        "the layout; see HIDDEN")
    return replace_initializer(model, "trunk_1_w", weight.T.copy())


def drop_the_normalizer(model):
    return bypass(model, {"obs_f64", "obs_centred", "obs_whitened",
                      "obs_clipped", "net_input"})


def normalize_twice(model):
    """A second copy of the whitening between the transform and the trunk.

    The failure of a deployer who whitens the observation themselves AND feeds
    it to a graph that whitens too, which is the other half of the reason the
    transform is inside the graph.
    """

    nodes = list(model.graph.node)
    chain = nodes[:5]
    assert [n.op_type for n in chain] == ["Cast", "Sub", "Div", "Clip", "Cast"]
    chain[-1].output[0] = "net_input_once"
    again = [
        helper.make_node("Cast", ["net_input_once"], ["again_f64"],
                         to=initializer(model, "obs_mean").data_type),
        helper.make_node("Sub", ["again_f64", "obs_mean"], ["again_centred"]),
        helper.make_node("Div", ["again_centred", "obs_std"], ["again_whitened"]),
        helper.make_node("Clip", ["again_whitened", "clip_lo", "clip_hi"],
                         ["again_clipped"]),
        helper.make_node("Cast", ["again_clipped"], ["net_input"],
                         to=initializer(model, "trunk_0_w").data_type),
    ]
    del model.graph.node[:]
    model.graph.node.extend(chain + again + nodes[5:])
    return model


def drop_the_normalizer_clip(model):
    """Only wrong outside the clip, which is what the far half of the
    observations exists to reach."""
    return bypass(model, {"obs_clipped"})


def drop_the_tanh(model):
    return bypass(model, {"squashed"})


def skip_the_action_rescale(model):
    """tanh straight onto the output: a unit-cube action on a +/-12 actuator."""
    return bypass(model, {"scaled", "action"})


def skip_the_bound_clip(model):
    """PPO's map is the clip, so dropping it is that policy's missing rescale."""
    return bypass(model, {"floored", "action"})


def drop_a_relu(model):
    return bypass(model, {"trunk_0_relu"})


def argmax_over_the_wrong_axis(model):
    for node in model.graph.node:
        if node.op_type == "ArgMax":
            del node.attribute[:]

            node.attribute.extend([helper.make_attribute("axis", 0),
                                   helper.make_attribute("keepdims", 0)])
    return model


def drop_the_legal_mask(model):
    """The plain ArgMax a graph writer who forgot the mask emits: a valid
    graph that commands actions the environment refuses."""
    return bypass(model, {"masked_row"})


def flatten_in_nchw_order(model):
    """The tower's features fed to the trunk channel-first, which is not the
    order the device's `Flatten` trained the first Linear on."""
    return bypass(model, {"conv_nhwc"})


def an_opset_that_predates_the_graph(model):
    """Clip moved its bounds from attributes to inputs at opset 11, and this
    graph clips. Declaring 10 describes a different Clip than it emits."""
    assert OPSET >= 11
    model.opset_import[0].version = 10
    return model


#: Every injection, the case it is applied to, and nothing else: the table is
#: the test. `goes_red` makes a graph that will not load, one the checker
#: rejects and one that computes the wrong action the same single column, so
#: an injection cannot pass by failing in a way this file forgot to look for.
INJECTIONS = [
    ("a trunk weight transposed", "ppo_continuous", transpose_a_weight),
    ("the normalizer dropped", "sac", drop_the_normalizer),
    ("the normalizer applied twice", "sac", normalize_twice),
    ("the normalizer's clip dropped", "sac", drop_the_normalizer_clip),
    ("tanh omitted", "sac", drop_the_tanh),
    ("the action rescale skipped", "sac", skip_the_action_rescale),
    ("a trunk ReLU dropped", "sac", drop_a_relu),
    ("the action bound clip skipped", "ppo_continuous", skip_the_bound_clip),
    ("argmax over the wrong axis", "ppo_discrete", argmax_over_the_wrong_axis),
    ("the legal-action mask dropped", "dqn_masked", drop_the_legal_mask),
    ("the conv features flattened NCHW", "dqn_image", flatten_in_nchw_order),
    ("an opset older than the graph's ops", "dqn",
     an_opset_that_predates_the_graph),
]


@pytest.mark.parametrize("what,case,injure", INJECTIONS,
                         ids=[i[0].replace(" ", "-") for i in INJECTIONS])
def test_an_injected_defect_goes_red(tmp_path, what, case, injure):
    policy, model, obs = build(tmp_path, case)
    assert _onnx.goes_red(policy, model, obs) is None, (
        f"the graph for {case} is not green before injecting {what!r}")
    before = model.SerializeToString()
    reason = _onnx.goes_red(policy, injure(model), obs)
    assert model.SerializeToString() != before, f"{what!r} changed nothing"
    print(f"\n{case}: {what} -> {reason}")
    assert reason is not None, f"{what!r} went unnoticed in {case}"
