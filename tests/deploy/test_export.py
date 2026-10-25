"""What `ttrl.export_policy` writes, and what it refuses to write.

Every refusal here is a silent-wrong-torque failure if it becomes a branch: a
dropped observation transform, an algorithm whose output map differs, a
privileged field the robot cannot sense. They are tested by their messages
because the message is what a roboticist acts on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.deploy import _checkpoints as discrete_fixture, _sac_checkpoint as fixture
import ttrl
from ttrl.algos.dqn.config import DQNConfig
from ttrl.deploy.read import META_KEY
from ttrl.deploy.policies import SOURCES, source_for
from ttrl.train.config import TrainConfig
from ttrl.spaces import ActionSpec, ObsField, ObsSpec

NORMALIZER = [(
    "NormalizeObservation",
    {"mean": np.array([1.0, -2.0, 0.5]), "m2": np.array([3.0, 12.0, 0.75]),
     "count": 4.0, "clip": 10.0, "epsilon": 1e-8},
)]


def export(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = fixture.write(tmp_path / "ckpt.npz", **kwargs)
    return ttrl.export_policy(source, str(tmp_path / "policy.npz"))


def meta(path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        return json.loads(npz[META_KEY].tobytes().decode("utf-8"))


def test_the_artifact_describes_the_policy_and_its_transform(tmp_path):
    out = export(tmp_path, env_policy_state=NORMALIZER)
    assert meta(out)["normalizer"]["count"] == 4.0
    policy = ttrl.load_policy(out)
    assert (policy.obs_dim, policy.action_dim) == (fixture.OBS_DIM,
                                                   fixture.ACTION_DIM)
    assert policy.normalizer["clip"] == 10.0
    np.testing.assert_array_equal(policy.action_high, [12.0, 4.0])


def test_refuses_a_checkpoint_that_does_not_say_what_transform_it_used(tmp_path):
    with pytest.raises(ValueError, match="carries no 'env_policy_state'"):
        export(tmp_path, env_policy_state=None)


def test_refuses_a_transform_it_does_not_reimplement(tmp_path):
    """A wrapper this format cannot apply would be dropped, which is exactly
    the silent failure the refusal exists for."""
    state = [("WhitenAndRotate", {"matrix": np.eye(3)})] + NORMALIZER
    with pytest.raises(ValueError, match=r"WhitenAndRotate"):
        export(tmp_path, env_policy_state=state)


def test_refuses_an_algorithm_it_holds_no_action_map_for():
    """An algorithm's weights only mean a policy through the map that turns
    its head's output into an action, so a config `ttrl.deploy.policies` does
    not name cannot be exported by guessing at one.

    Called directly rather than through a checkpoint because a checkpoint may
    only name a type inside `ttrl`, so a third party's config cannot be
    written into one at all.
    """
    with pytest.raises(ValueError) as caught:
        source_for(TrainConfig(), "p.npz")
    message = str(caught.value)
    assert "TrainConfig" in message
    for source in SOURCES:
        assert source.config.__name__ in message


def test_refuses_an_action_kind_its_algorithm_has_no_map_for(tmp_path):
    """SAC's policy is a squashed Gaussian over a box. Handed a discrete
    action spec its mean head's row is not logits over anything, so the
    combination is refused rather than argmaxed."""
    with pytest.raises(ValueError, match="no action map"):
        export(tmp_path, env_policy_state=NORMALIZER,
               action_spec=ActionSpec.discrete(4))


def test_a_masked_policy_exports_and_never_chooses_an_illegal_action(tmp_path):
    """The mask is the tail of the observation, so the artifact carries its
    width and the deployed argmax runs over the legal columns only.

    The gate is on the rows where masking CHANGES the choice: an artifact that
    dropped the mask agrees with this one everywhere else, so a batch without
    those rows would pass for an exporter that ignored it.
    """

    actions, obs_dim = 3, 8
    spec = ActionSpec.discrete(actions)
    masked = ObsSpec.vector(obs_dim, action_mask=actions)
    path = export(tmp_path, env_policy_state=[], obs_spec=masked,
                  action_spec=spec,
                  state=discrete_fixture.dqn_state(spec, obs_dim=obs_dim))
    policy = ttrl.load_policy(path)
    assert policy.action_mask == actions

    rng = np.random.default_rng(3)
    features = rng.normal(size=(64, obs_dim - actions)).astype(np.float32)
    # Every legal set with at least one action, so some rows forbid the
    # unmasked argmax and the rest leave it alone.
    sets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                     [0, 1, 1], [1, 1, 1]], np.float32)
    masks = sets[rng.integers(0, len(sets), 64)]
    obs = np.concatenate([features, masks], axis=1)

    unmasked = np.argmax(policy.head(obs), axis=-1)
    chosen = policy(obs)
    legal = masks[np.arange(64), chosen] > 0.5
    changed = chosen != unmasked
    assert legal.all(), "deployed policy returned an action its own mask forbade"
    assert changed.sum() >= 5, "no row in this batch exercises the mask"
    # On every other row the mask leaves the unmasked argmax legal, so the
    # masked choice must be the same action, not merely a legal one.
    assert np.array_equal(chosen[~changed], unmasked[~changed])
    np.testing.assert_array_equal(policy.legal_mask(obs), masks)


def test_refuses_a_mask_over_a_continuous_action(tmp_path):
    """A mask names members of a finite action set, and SAC's squashed
    Gaussian chooses no column for it to name."""
    with pytest.raises(ValueError, match="finite action set"):
        export(tmp_path, env_policy_state=[],
               obs_spec=ObsSpec.vector(fixture.OBS_DIM, action_mask=2))


def test_refuses_a_mask_the_normaliser_would_whiten(tmp_path):
    """Whitening rewrites the mask's indicators out of {0, 1}, and the
    deployed reader reads them back as a mask."""

    spec = ActionSpec.discrete(2)
    with pytest.raises(ValueError, match="whiten the mask"):
        export(tmp_path, env_policy_state=NORMALIZER,
               obs_spec=ObsSpec.vector(fixture.OBS_DIM, action_mask=2),
               action_spec=spec, state=discrete_fixture.dqn_state(spec))


def test_refuses_weights_whose_width_is_not_the_actor_s(tmp_path):
    """A spec that reserves a field for the critic alone and weights that read
    the whole observation disagree about what the actor saw."""
    spec = ObsSpec.split_vector(
        ObsField("sensed", fixture.OBS_DIM - 1),
        ObsField("privileged", 1, actor=False),
    )
    with pytest.raises(ValueError, match="privileged"):
        export(tmp_path, env_policy_state=NORMALIZER, obs_spec=spec)


def test_the_observation_scale_is_folded_into_the_first_weight(tmp_path):
    """`ScaledLinear` folds `ObsSpec.scale` into the weight rather than
    multiplying the observation, so the artifact folds it the same way and
    the two forwards are the same ops in the same order."""

    half = ObsSpec(shape=(fixture.OBS_DIM,), dtype="float32", kind="vector",
                   scale=0.5)
    scaled = ttrl.load_policy(export(tmp_path / "a", env_policy_state=[],
                                     obs_spec=half))
    plain = ttrl.load_policy(export(tmp_path / "b", env_policy_state=[]))
    obs = np.array([0.4, -1.1, 2.0], np.float32)
    np.testing.assert_allclose(scaled(obs), plain(obs * 0.5), rtol=1e-6)


def test_a_checkpoint_handed_to_the_artifact_reader_is_named_as_one(tmp_path):
    """A checkpoint is an npz with a `__meta__` member too, so the obvious
    mistake gets the obvious answer rather than an artifact version of None."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    source = fixture.write(tmp_path / "ckpt.npz", env_policy_state=NORMALIZER)
    with pytest.raises(ValueError, match="is a ttrl CHECKPOINT"):
        ttrl.load_policy(source)


def test_refuses_an_image_encoder_whose_strides_it_cannot_read(tmp_path):
    """A checkpoint stores conv weights but not their strides, so a tower the
    exporter cannot name would be exported with guessed geometry."""
    state = discrete_fixture.dqn_image_state()
    state["config"] = DQNConfig(encoder="my_tower")
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = fixture.write(tmp_path / "ckpt.npz", env_policy_state=[], state=state,
                           obs_spec=discrete_fixture.IMAGE_SPEC,
                           action_spec=discrete_fixture.DISCRETE)
    with pytest.raises(ValueError, match="'my_tower'"):
        ttrl.export_policy(source, str(tmp_path / "policy.npz"))
