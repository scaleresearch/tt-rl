"""SAC's Polyak target sync: the blend, its address stability, and the name
pairing, eager and replayed from a trace.

Split out of `test_algos_sac_parity.py`, which is the float64 torch-twin gate
on the three losses. Nothing here needs torch: the blend is two multiplies and
an add whose reference is one line of numpy, and what is worth asserting is
the mechanism around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttrl.algos.sac import SAC, SACConfig
from ttrl.algos.sac.targets import PolyakSync
from ttrl.engine import acquire
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(17))
ACTIONS = ActionSpec.continuous([-1.0] * 4, [1.0] * 4)
BATCH = 64


@pytest.fixture(scope="module")
def device():
    with acquire(0) as (dev, _profile):
        yield dev


@pytest.fixture
def agent(device):
    cfg = SACConfig(batch_size=BATCH, learning_starts=BATCH, tau=0.01)
    built = SAC(OBS, ACTIONS, device, cfg=cfg, n_envs=1, seed=13)
    try:
        yield built
    finally:
        built.close()


@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
def test_polyak_blends_each_target_towards_its_own_critic(agent, device, traced):
    """target <- tau * critic + (1 - tau) * target, in place and by name.

    Two rounds with the critics moved in between, because a captured blend
    that replayed the capture pass's values rather than the current ones
    would pass a single round.
    """
    sync = PolyakSync(agent.nets, agent.cfg.tau, device, traced=traced)
    addresses = {name: id(param.data)
                 for name, param in agent.nets.targets.named_parameters()}
    rng = np.random.default_rng(0)
    tau = agent.cfg.tau
    try:
        for round_index in range(2):
            # Move the critics away from the targets, or the blend is a no-op.
            agent.nets.critics.load_state_dict(
                {name: array + rng.normal(scale=0.5, size=array.shape).astype(np.float32)
                 for name, array in agent.nets.critics.state_dict().items()}, device)
            before = {name: param.numpy().copy()
                      for name, param in agent.nets.targets.named_parameters()}
            sources = {name: param.numpy().copy()
                       for name, param in agent.nets.critics.named_parameters()}

            sync.run()

            worst = 0.0
            for name, param in agent.nets.targets.named_parameters():
                want = (1.0 - tau) * before[name] + tau * sources[name]
                worst = max(worst, float(np.abs(param.numpy() - want).max()))
                assert id(param.data) == addresses[name], (
                    f"polyak rebound {name}'s buffer; a captured trace would "
                    f"keep replaying against the old address"
                )
            assert worst < 1e-6, f"round {round_index}: worst {worst}"
    finally:
        sync.close()


def test_polyak_checks_the_name_pairing_on_every_call(agent, device):
    """The pairing check is host-side and outside the captured region, so a
    structure that changes after capture is still caught."""
    sync = PolyakSync(agent.nets, agent.cfg.tau, device, traced=True)
    target = list(agent.nets.targets)[0]
    original = target.named_parameters
    try:
        sync.run()
        sync.run()

        def renamed():
            return [(f"wrong.{name}", param) for name, param in original()]

        target.named_parameters = renamed
        with pytest.raises(ValueError, match="cross-wire"):
            sync.run()
    finally:
        target.named_parameters = original
        sync.close()
