"""The deployed policy's arithmetic, against the code it has to agree with.

The transform is checked against `NormalizeObservation` itself rather than
against a second copy of Welford's formula written here, which would only
prove that two spellings of the same mistake agree. The trick is that the
wrapper normalises an observation with the statistics that already include
it, so a snapshot taken immediately AFTER a step describes exactly the
transform that step's output went through.

The wrapper lives above the vector environment, because its estimator is one
object over every lane, so the stack here is one over a one-lane
`SyncVectorEnv` and the observation compared is that lane's.

The network's arithmetic against the real device actor is
`tests/hardware/test_deploy_parity.py`; this file pins everything that needs
no chip.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tests._fakes.toy_envs import ReachEnv
from tests.deploy import _sac_checkpoint as fixture
import ttrl
from ttrl.algos.sac.config import SACConfig
from ttrl.envs.vec_normalize import NormalizeObservation
from ttrl.envs.vector import SyncVectorEnv
from ttrl.envs.wrappers import Wrapper
from ttrl.spaces import DEFAULT_LEAF, ActionSpec, ObsField, ObsSpec

BLOCKED = ("ttnn", "gymnasium", "mujoco", "cv2", "ale_py", "ttrl.engine",
           "ttrl.algos", "ttrl.train", "ttrl.envs", "ttrl.nets")


class Tap(Wrapper):
    """Remembers the observation it passes up, which is the robot's input."""

    POLICY_STATE = ()

    def __init__(self, env):
        super().__init__(env)
        self.last = None

    def reset(self, *, seed=None):
        self.last, info = self.env.reset(seed=seed)
        return self.last, info

    def step(self, action):
        self.last, reward, terminated, truncated, info = self.env.step(action)
        return self.last, reward, terminated, truncated, info


def stepped(steps: int, *, clip: float = 10.0):
    """`(raw observation, the wrapper's output for it, the state right after)`."""
    tap = Tap(ReachEnv(seed=3))
    env = NormalizeObservation(SyncVectorEnv([lambda: tap]), clip=clip)
    whitened, _info = env.reset(seed=3)
    for _ in range(steps):
        whitened, *_rest = env.step(np.zeros((1, 2), np.float32))
    return tap.last, whitened[DEFAULT_LEAF][0], env.save_policy_state()


def policy_for(tmp_path, policy_state, *, obs_spec=None, seed=0):
    source = fixture.write(tmp_path / "ckpt.npz", env_policy_state=policy_state,
                           obs_spec=obs_spec or fixture.OBS_SPEC, seed=seed)
    return ttrl.load_policy(ttrl.export_policy(source, str(tmp_path / "p.npz")))


@pytest.mark.parametrize("steps", [0, 17])
def test_normalize_reproduces_the_training_wrapper_bit_for_bit(tmp_path, steps):
    """Not `allclose`: the deployed controller's input must be the input the
    weights were fitted on."""
    raw, whitened, state = stepped(steps)
    policy = policy_for(tmp_path, state)
    np.testing.assert_array_equal(policy.normalize(raw), whitened)


def test_statistics_are_frozen_in_the_artifact(tmp_path):
    """Deployment must not keep fitting: two robots started from one file
    compute the same torque for the same sensor reading, forever."""
    raw, _whitened, state = stepped(17)
    policy = policy_for(tmp_path, state)
    first = policy.normalize(raw)
    for _ in range(50):
        policy(np.full(fixture.OBS_DIM, 30.0, np.float32))
    np.testing.assert_array_equal(policy.normalize(raw), first)


def test_only_the_actor_s_columns_of_the_transform_are_carried(tmp_path):
    """A critic-only field is state the robot cannot sense, so neither its
    statistics nor its input column appear in the artifact."""
    spec = ObsSpec.split_vector(
        ObsField("sensed", fixture.OBS_DIM, actor=True),
        ObsField("privileged", 2, actor=False),
    )
    _raw, _whitened, state = stepped(9)
    wide = [(state[0][0], {**state[0][1],
                           "mean": np.pad(state[0][1]["mean"], (0, 2)),
                           "m2": np.pad(state[0][1]["m2"], (0, 2))})]
    policy = policy_for(tmp_path, wide, obs_spec=spec)
    assert policy.obs_dim == fixture.OBS_DIM
    np.testing.assert_array_equal(
        policy.normalize(_raw), policy_for(tmp_path, state).normalize(_raw))


def test_the_action_stays_inside_the_actuator_bounds(tmp_path):
    """tanh times the half width plus the centre cannot leave the box, which
    is the one property a robot's safety case needs from this file."""
    _raw, _whitened, state = stepped(9)
    policy = policy_for(tmp_path, state)
    batch = np.random.default_rng(0).normal(scale=50.0,
                                            size=(256, fixture.OBS_DIM))
    actions = policy(batch.astype(np.float32))
    assert actions.shape == (256, fixture.ACTION_DIM)
    assert (actions >= policy.action_low).all()
    assert (actions <= policy.action_high).all()


def test_a_robot_host_runs_the_artifact_with_numpy_alone(tmp_path):
    """The point of the whole artifact: no chip, no engine, no env suite, and
    no part of the training stack on the machine that commands the robot."""
    assert _in_a_bare_process("import ttnn").returncode != 0, "the blocker blocks nothing"
    _raw, _whitened, state = stepped(9)
    policy_for(tmp_path, state)  # written by the training-side import
    result = _in_a_bare_process(f"""
        from ttrl.deploy.read import load_policy
        policy = load_policy({str(tmp_path / 'p.npz')!r})
        print(policy(numpy.zeros(policy.obs_dim, numpy.float32)).shape)
        print(sorted(m for m in sys.modules if m.startswith("ttrl.")))
        """)
    assert result.returncode == 0, result.stderr
    shape, modules = result.stdout.strip().splitlines()
    assert shape == f"({fixture.ACTION_DIM},)"
    assert modules == ("['ttrl.deploy', 'ttrl.deploy.artifact', 'ttrl.deploy.conv', "
                       "'ttrl.deploy.read']")


def _in_a_bare_process(body: str) -> subprocess.CompletedProcess:
    blocker = f"""
        import sys
        import numpy


        class Blocked(Exception):
            pass


        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {BLOCKED!r} or fullname.split(".")[0] in {BLOCKED!r}:
                    raise Blocked(f"{{fullname}} was imported")
                return None


        sys.meta_path.insert(0, Blocker())
        """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(blocker) + textwrap.dedent(body)],
        capture_output=True, text=True,
    )


def test_the_trunk_rectifies(tmp_path):
    """A trunk whose ReLUs went missing still runs, still stays in bounds and
    still looks like a policy: what it loses is every nonlinearity. One
    scalar layer of weight 1 and no bias makes the expected action arithmetic
    rather than a second copy of the forward pass.
    """

    one = np.ones((1, 1), np.float32)
    state = {"config": SACConfig(), "weights_only": True,
             "log_alpha": one.copy(),
             "nets": {f"policy.encoder.towers.{DEFAULT_LEAF}.layers.0.weight": one.copy(),
                      f"policy.encoder.towers.{DEFAULT_LEAF}.layers.0.bias":
                          np.zeros((1, 1), np.float32),
                      "policy.head.heads.mean.weight": one.copy(),
                      "policy.head.heads.mean.bias": np.zeros((1, 1), np.float32)}}
    source = fixture.write(tmp_path / "ckpt.npz", env_policy_state=[], state=state,
                           obs_spec=ObsSpec.vector(1),
                           action_spec=ActionSpec.continuous([-10.0], [10.0]))
    policy = ttrl.load_policy(ttrl.export_policy(source, str(tmp_path / "p.npz")))

    np.testing.assert_allclose(policy(np.array([0.7], np.float32)),
                               10.0 * np.tanh(0.7), rtol=1e-6)
    # Rectified to exactly zero, which tanh then maps to the centre of the box.
    np.testing.assert_array_equal(policy(np.array([-0.7], np.float32)), [0.0])
