"""One DQN run's exact bytes, in every step mode, against a recorded digest.

TWO README CLAIMS LIVE HERE.

  "divergence 0.0 on every parameter against the untraced path" is the parity
  claim. `test_every_step_mode_agrees_bit_for_bit` runs the same seeded agent
  once per mode in `ttrl.engine.step.MODES` and requires one distinct
  fingerprint. The modes are enumerated from the engine, so a third mode is
  covered the day it is added rather than the day somebody remembers this
  file.

  Cross-process determinism is the other, and `RECORDED` is where it lives: a
  digest measured in a different process on a different day. A process that
  reproduces it reproduced every float of an eight-update DQN run, which is
  what "bit-identical across two processes" means. Spawning the second process
  from the test cannot work here, because ttnn keeps its /dev/tenstorrent
  descriptors for the life of a process, so `acquire` in a child of a pytest
  process that has opened a chip is refused rather than served;
  `tests/engine/test_device_stale_fd.py` asserts that on purpose. A digest
  crosses the process boundary instead, and crosses time with it.

  The measurement that produced `RECORDED` is the `__main__` block at the
  bottom of this file, so the harness cannot drift away from the number it
  produced:

      TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=... \
        python -m tests.hardware.test_determinism_fingerprint 2

  It holds no device itself and runs each child in turn, which is the only
  arrangement in which a child can acquire a chip.

WHAT MAKES THE DIGEST MOVE. Any change to the ops, the autograd tape, the Adam
step, the trace path, the replay sampler or the network builder, and a ttnn or
driver version bump. That is the point: a bit-exactness claim is only a gate if
every one of those has to be re-measured deliberately. When a change is
intended, re-run `python -m pytest` on this file, read the digest out of the
failure message, and record the new one with a note saying what moved it.
Recording a digest nobody measured in a second process is the one thing that
would make this file a lie.

Device shapes are small on purpose. The claim is about bit-exactness, which a
32-row vector observation tests as sharply as an 84x84 frame, and the Nature
CNN's own trace determinism is `tests/hardware/test_conv_trace_determinism.py`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from ttrl.algos.base import Transition
from ttrl.algos.dqn import DQN, DQNConfig
from ttrl.algos.sac import SAC, SACConfig
from ttrl.engine import acquire
from ttrl.engine.step import MODES
from ttrl.spaces import ActionSpec, ObsGroup, ObsSpec

pytestmark = pytest.mark.tt

OBS = ObsGroup.of(ObsSpec.vector(8))
ACTIONS = ActionSpec.discrete(4)
LANES = 2
SEED = 20261023
#: Frames fed before any update, and updates measured after them. Eight
#: updates is enough that the Adam moments, the target sync and the replay
#: sampler have all moved, and short enough to run in seconds in each mode.
FILL_FRAMES = 128
UPDATES = 8

#: The digest `fingerprint` produced, measured on Blackhole with ttnn 0.80.0 in
#: two independent processes on 2026-10-23, each scoped to one chip, in both
#: step modes: four runs, one digest. The launcher held no device, which is
#: what let each child acquire one; see this module's docstring.
#:
#: Moved once, on 2026-10-23, by observations becoming their named leaves: a
#: parameter the digest covers by NAME is now `encoder.towers.obs.layers.0.
#: weight` where it was `encoder.layers.0.weight`. No float moved, and that
#: was measured rather than assumed: recomputing this digest with the
#: `towers.obs.` segment deleted from every name reproduced the previous
#: digest a377056ff63bf0af69c1ae017068c451e254dea9db483f93edc2196cb4d7ef1e
#: exactly, in both step modes, over the same eight-update run (loss
#: 1.22304738 -> 0.66329253 before and after).
RECORDED = "427a7979dea2a40b717d13b2cd75e2254fa327cb2e9d0cf80be1351e04b61757"


def _cfg(mode: str) -> DQNConfig:
    return DQNConfig(batch_size=32, learning_starts=64, train_freq=2, n_step=2,
                     gamma=0.95, lr=1e-3, target_sync_interval=3,
                     eps_decay_frames=512, buffer_capacity=1024,
                     step_mode=mode)


def _transitions(agent, rng) -> None:
    """Feed deterministic synthetic transitions. No environment, because an
    env is one more thing whose own determinism this would be asserting."""
    obs = rng.standard_normal((LANES, 8)).astype(np.float32)
    for frame in range(0, FILL_FRAMES, LANES):
        action = np.arange(LANES, dtype=np.int64) + frame % ACTIONS.n
        action %= ACTIONS.n
        next_obs = rng.standard_normal((LANES, 8)).astype(np.float32)
        reward = rng.standard_normal(LANES).astype(np.float32)
        terminated = (np.arange(LANES) + frame) % 17 == 0
        truncated = np.zeros(LANES, dtype=bool)
        info = {"final_obs": [next_obs[i] if terminated[i] else None
                              for i in range(LANES)]}
        agent.observe(Transition.from_step(
            OBS, obs, action,
            (next_obs, reward, terminated, truncated, info)))
        obs = next_obs


def fingerprint(device, mode: str) -> tuple[str, list[dict[str, float]]]:
    """(digest, per-update metrics) for one seeded DQN run in `mode`."""
    agent = DQN(OBS, ACTIONS, device, cfg=_cfg(mode), n_envs=LANES, seed=SEED)
    try:
        _transitions(agent, np.random.default_rng(SEED))
        metrics = [agent.update() for _ in range(UPDATES)]
        digest = hashlib.sha256()
        for update in metrics:
            for key in sorted(update):
                digest.update(key.encode())
                digest.update(np.float64(update[key]).tobytes())
        for name, param in agent.online.named_parameters():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(param.numpy(), np.float32).tobytes())
        return digest.hexdigest(), metrics
    finally:
        agent.close()


# -- SAC ---------------------------------------------------------------------

#: SAC's own eager-vs-traced claim. There is no recorded cross-process digest
#: for it: that claim is DQN's and is about the engine, which both algorithms
#: share. What is SAC's alone is that its two regions, their three optimizers
#: and the Polyak sync replay to the same bytes they compute eagerly, and that
#: is what the test below asserts. The algorithm itself is gated against a
#: float64 torch twin in `tests/hardware/test_algos_sac_parity.py`.
SAC_ACTIONS = ActionSpec.continuous([-1.0] * 3, [1.0] * 3)
SAC_BATCH = 32


def _sac_cfg(mode: str) -> SACConfig:
    return SACConfig(batch_size=SAC_BATCH, learning_starts=SAC_BATCH,
                     buffer_capacity=1024, train_freq=2, n_step=2, gamma=0.95,
                     lr=1e-3, tau=0.05, init_alpha=0.3, step_mode=mode)


def _sac_transitions(agent, rng) -> None:
    obs = rng.standard_normal((LANES, 8)).astype(np.float32)
    for frame in range(0, FILL_FRAMES, LANES):
        action = rng.uniform(-1.0, 1.0, (LANES, SAC_ACTIONS.dim)).astype(np.float32)
        next_obs = rng.standard_normal((LANES, 8)).astype(np.float32)
        reward = rng.standard_normal(LANES).astype(np.float32)
        terminated = (np.arange(LANES) + frame) % 17 == 0
        info = {"final_obs": [next_obs[i] if terminated[i] else None
                              for i in range(LANES)]}
        agent.observe(Transition.from_step(
            OBS, obs, action,
            (next_obs, reward, terminated, np.zeros(LANES, dtype=bool), info)))
        obs = next_obs


def sac_fingerprint(device, mode: str) -> tuple[str, list[dict[str, float]]]:
    """(digest, per-update metrics) for one seeded SAC run in `mode`."""
    agent = SAC(OBS, SAC_ACTIONS, device, cfg=_sac_cfg(mode), n_envs=LANES,
                seed=SEED)
    try:
        _sac_transitions(agent, np.random.default_rng(SEED))
        metrics = [agent.update() for _ in range(UPDATES)]
        digest = hashlib.sha256()
        for update in metrics:
            for key in sorted(update):
                digest.update(key.encode())
                digest.update(np.float64(update[key]).tobytes())
        for name, array in sorted(agent.nets.state_dict().items()):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array, np.float32).tobytes())
        digest.update(np.ascontiguousarray(agent.log_alpha.numpy(),
                                           np.float32).tobytes())
        return digest.hexdigest(), metrics
    finally:
        agent.close()


#: Each covered algorithm's fingerprint function and the loss key its metrics
#: carry. `tests/test_algos_step_modes.py` holds the same set and fails the
#: day a third algorithm declares a step mode.
ALGOS = {"dqn": (fingerprint, "loss"), "sac": (sac_fingerprint, "critic_loss")}


@pytest.fixture(scope="module")
def prints(request):
    """{algo: {mode: (digest, metrics)}}, every algorithm on one device."""
    with acquire(0) as (device, _profile):
        out = {algo: {mode: fn(device, mode) for mode in MODES}
               for algo, (fn, _key) in ALGOS.items()}
    for algo, (_fn, key) in ALGOS.items():
        for mode, (digest, metrics) in out[algo].items():
            print(f"\n{algo} {mode}: {digest}  {key} {metrics[0][key]:.8f} -> "
                  f"{metrics[-1][key]:.8f}")
    return out


def test_the_run_actually_trained(prints):
    """A fingerprint of a run that did nothing would agree with itself."""
    for algo, (_fn, key) in ALGOS.items():
        for mode, (_digest, metrics) in prints[algo].items():
            losses = [m[key] for m in metrics]
            assert len(metrics) == UPDATES
            assert all(np.isfinite(v) for m in metrics for v in m.values()), metrics
            assert len(set(losses)) == UPDATES, (
                f"{algo} {mode}: {len(set(losses))} distinct {key} values in "
                f"{UPDATES} updates, so the weights are not moving and this "
                f"file asserts nothing"
            )


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_every_step_mode_agrees_bit_for_bit(prints, algo):
    digests = {mode: digest for mode, (digest, _m) in prints[algo].items()}
    assert len(MODES) >= 2, MODES
    assert len(set(digests.values())) == 1, (
        f"{algo}: the step modes disagree: {digests}. The traced path and the "
        f"eager path must produce identical bytes; the README's 'divergence "
        f"0.0 on every parameter against the untraced path' is this."
    )


def test_the_fingerprint_matches_the_recorded_one(prints):
    digest = next(iter(prints["dqn"].values()))[0]
    assert digest == RECORDED, (
        f"this run fingerprints as {digest}, the recorded cross-process "
        f"measurement is {RECORDED}. Either something on the numeric path "
        f"changed, or the installed ttnn or driver did. Re-measure in two "
        f"independent processes before recording a new digest; see this "
        f"module's docstring."
    )


#: What each child of `_measure` runs: one process, one chip, every mode.
_CHILD = """
import json, sys
sys.path.insert(0, ".")
from tests.hardware.test_determinism_fingerprint import fingerprint
from ttrl.engine import acquire
from ttrl.engine.step import MODES

out = {}
with acquire(0) as (device, _profile):
    for mode in MODES:
        out[mode] = fingerprint(device, mode)[0]
print("RESULT " + json.dumps(out))
"""


def _measure(procs: int) -> int:
    """Fingerprint in `procs` independent processes, every mode in each.

    The launcher opens no device, which is what lets each child acquire one.
    Prints every digest seen and exits non-zero unless there is exactly one.
    """

    seen: dict[str, list[str]] = {}
    for i in range(procs):
        done = subprocess.run([sys.executable, "-c", _CHILD],
                              capture_output=True, text=True, timeout=900)
        found = [line for line in done.stdout.splitlines()
                 if line.startswith("RESULT ")]
        if not found:
            print(done.stdout[-2000:], done.stderr[-4000:])
            return 1
        for mode, digest in json.loads(found[0][len("RESULT "):]).items():
            seen.setdefault(digest, []).append(f"process {i} {mode}")
    print(f"{len(seen)} distinct fingerprint(s) over {procs} processes:")
    for digest, who in seen.items():
        print(f"  {digest}  {', '.join(who)}")
    return 0 if len(seen) == 1 else 2


if __name__ == "__main__":
    import sys

    sys.exit(_measure(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
