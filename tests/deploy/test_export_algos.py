"""Every algorithm the library registers exports, and exports its own policy.

THE SET IS DERIVED, NOT TYPED. `ALGOS.names()` is the library's answer to
"which algorithms are there", and the claim here is that
`ttrl.deploy.policies.SOURCES` covers exactly that set, and that every action
map a source names is one `ttrl.deploy.artifact` implements. Registering a
fourth algorithm fails this file until its actor is placed and its action map
chosen, which is the point: an exporter that silently covered two thirds of
the registry is how a trained policy turns out to be unexportable on the day
someone needs it off the box.

What each algorithm's deterministic action IS is the other half, and it is
checked by arithmetic rather than by name: the same weights through SAC's
squash and through PPO's clip are two different functions, and if the
exporter picked one map for both, one of the two algorithms would deploy a
policy nobody trained.
"""

from __future__ import annotations


import numpy as np
import pytest

from tests.deploy import _checkpoints as fixture
from tests.deploy import _sac_checkpoint as sac_fixture
import ttrl
from ttrl.algos.registry import ALGOS
from ttrl.deploy.artifact import ACTION_MAPS
from ttrl.deploy.read import META_KEY
from ttrl.deploy.policies import SOURCES


NORMALIZER = [(
    "NormalizeObservation",
    {"mean": np.array([1.0, -2.0, 0.5]), "m2": np.array([3.0, 12.0, 0.75]),
     "count": 4.0, "clip": 10.0, "epsilon": 1e-8},
)]

#: One host-built agent state per registered algorithm. Keyed by the
#: registry's own name so a registered algorithm with no fixture here is a
#: missing entry rather than a test nobody wrote.
STATES = {
    "sac": lambda spec, seed: sac_fixture.agent_state(seed=seed),
    "ppo": lambda spec, seed: fixture.ppo_state(spec, seed=seed),
    "dqn": lambda spec, seed: fixture.dqn_state(spec, seed=seed),
}

SPECS = {"continuous": fixture.CONTINUOUS, "discrete": fixture.DISCRETE}

#: Every (algorithm, action kind) this exporter claims, derived from the table.
COVERED = [(source.algo, kind) for source in SOURCES for kind in source.maps]


def export(tmp_path, algo: str, kind: str, *, seed: int = 0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = SPECS[kind]
    source = sac_fixture.write(
        tmp_path / "ckpt.npz", env_policy_state=NORMALIZER,
        state=STATES[algo](spec, seed), obs_spec=fixture.OBS_SPEC,
        action_spec=spec)
    return ttrl.export_policy(source, str(tmp_path / "policy.npz"))


def test_every_registered_algorithm_has_a_policy_source():
    assert sorted(s.algo for s in SOURCES) == sorted(ALGOS.names())


def test_every_registered_algorithm_has_a_host_built_state():
    """The fixtures this file exports from cover the registry too, so a new
    algorithm cannot be declared covered by an entry nothing exercises."""
    assert sorted(STATES) == sorted(ALGOS.names())


def test_every_action_map_a_source_names_is_one_the_reader_implements():
    named = {m for source in SOURCES for m in source.maps.values()}
    assert named == set(ACTION_MAPS)


@pytest.mark.parametrize("algo,kind", COVERED, ids=[f"{a}-{k}" for a, k in COVERED])
def test_the_policy_exports_and_acts(tmp_path, algo, kind):
    policy = ttrl.load_policy(export(tmp_path, algo, kind))
    assert (policy.algo, policy.action_kind) == (algo, kind)
    obs = np.array([0.3, -1.2, 4.0], np.float32)
    action = policy(obs)
    batch = policy(np.stack([obs, obs * -1.0]))
    if kind == "discrete":
        assert action.dtype == np.int64 and action.shape == ()
        assert batch.shape == (2,)
        assert 0 <= int(action) < fixture.N_ACTIONS
    else:
        assert action.dtype == np.float32
        assert action.shape == (fixture.ACTION_DIM,) and batch.shape == (2, 2)
        assert np.all(action >= policy.action_low)
        assert np.all(action <= policy.action_high)


def test_sacs_squash_and_ppos_clip_are_two_functions(tmp_path):
    """The one bug that cannot be caught by looking at the artifact.

    SAC's deterministic action is `tanh(mean)` mapped onto the bounds; PPO's
    continuous policy clips its sample to them instead. Handed the same trunk
    and the same head, those disagree, so the map the exporter chose is
    load-bearing and not a label.
    """

    shared = sac_fixture.sac_nets(seed=3)
    ppo_state = fixture.ppo_state(fixture.CONTINUOUS, seed=3)
    for index in range(2):
        for part in ("weight", "bias"):
            key = f"layers.{2 * index}.{part}"
            ppo_state["net"][f"encoders.actor.towers.obs.{key}"] = \
                shared[f"policy.encoder.towers.obs.{key}"]
    for part in ("weight", "bias"):
        ppo_state["net"][f"heads.policy.heads.mean.{part}"] = \
            shared[f"policy.head.heads.mean.{part}"]

    def artifact(name, state):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        source = sac_fixture.write(
            directory / "ckpt.npz", env_policy_state=NORMALIZER, state=state,
            obs_spec=fixture.OBS_SPEC, action_spec=fixture.CONTINUOUS)
        return ttrl.load_policy(
            ttrl.export_policy(source, str(directory / "policy.npz")))

    squashed = artifact("sac", sac_fixture.agent_state(seed=3))
    clipped = artifact("ppo", ppo_state)
    assert squashed.action_map == "tanh_bounds"
    assert clipped.action_map == "clip_bounds"
    obs = (np.random.default_rng(1).normal(size=(64, 3)) * 3.0).astype(np.float32)
    np.testing.assert_allclose(squashed.head(obs), clipped.head(obs), rtol=1e-6)
    gap = float(np.abs(squashed(obs) - clipped(obs)).max())
    assert gap > 1.0, f"the two maps agree to {gap} on the same weights"


def test_the_exporter_reads_the_actor_and_not_a_critic_or_a_target(tmp_path):
    """Every fixture state carries the parameters the artifact must not take.

    PPO's critic trunk and value head and DQN's lagged target are the same
    shapes as the actor's, so an exporter matching by shape rather than by
    anchored name would export one of them.
    """

    out = export(tmp_path / "ppo", "ppo", "discrete")
    with np.load(out, allow_pickle=False) as npz:
        exported = {k: npz[k] for k in npz.files if k != META_KEY}
    state = fixture.ppo_state(fixture.DISCRETE)["net"]
    np.testing.assert_array_equal(
        exported["trunk.0.weight"],
        state["encoders.actor.towers.obs.layers.0.weight"])
    assert not np.array_equal(
        exported["trunk.0.weight"],
        state["encoders.critic.towers.obs.layers.0.weight"])

    out = export(tmp_path / "dqn", "dqn", "discrete")
    with np.load(out, allow_pickle=False) as npz:
        trunk = npz["trunk.0.weight"]
    dqn = fixture.dqn_state(fixture.DISCRETE)
    np.testing.assert_array_equal(
        trunk, dqn["online"]["encoder.towers.obs.layers.0.weight"])
    assert not np.array_equal(
        trunk, dqn["target"]["encoder.towers.obs.layers.0.weight"])


def test_a_recurrent_ppo_policy_is_refused_rather_than_exported_without_its_gru(
        tmp_path):
    """`RecurrentActorCritic` names its trunk and heads as `ActorCritic` does,
    so every pattern the exporter reads matches and only the `memories.*`
    parameters tell the two apart. Without the refusal the artifact is the
    trunk and head with the GRU cut out of the middle, which loads and acts."""

    state = fixture.ppo_state(fixture.DISCRETE)
    rng = np.random.default_rng(7)
    width = fixture.HIDDEN[-1]
    for name, shape in (("weight_ih", (width, 3 * width)),
                        ("weight_hh", (width, 3 * width)),
                        ("bias_ih", (1, 3 * width)),
                        ("bias_hh", (1, 3 * width))):
        state["net"][f"memories.actor.cell.{name}"] = \
            rng.standard_normal(shape).astype(np.float32)
    source = sac_fixture.write(
        tmp_path / "ckpt.npz", env_policy_state=NORMALIZER, state=state,
        obs_spec=fixture.OBS_SPEC, action_spec=fixture.DISCRETE)
    with pytest.raises(ValueError, match="Recurrent export is unsupported"):
        ttrl.export_policy(source, str(tmp_path / "policy.npz"))
    assert not (tmp_path / "policy.npz").exists()
