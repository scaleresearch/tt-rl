"""Running an exported ONNX graph, and measuring it against the artifact.

THE COMPARISON IS AGAINST THE NUMPY ARTIFACT. Both sides run on this host's
CPU in fp32 from the same stored weights, so the only thing between them is
the order a matmul accumulates in. That makes this gate far tighter than the
device gate it sits beside: `tests/hardware/test_deploy_parity.py` allows
0.1 Nm on a +/-12 Nm actuator because the chip's matmul accumulates through
HiFi4 and numpy's does not, a floor around 1.3e-3 relative. onnxruntime on
host CPU is MORE accurate than the chip, not less, so nothing here needs that
much room, and leaving it would let a real defect through.

`parity` reports ONE number per policy kind, scaled so a tolerance reads the
same for every action space:

  continuous   the largest action deviation as a fraction of the action's
               half-width, which is the deviation a joint would see divided
               by the torque it is allowed.
  discrete     the fraction of observations whose chosen action differs.
               Exactness is the only honest bar for an index, and
               `min_decision_gap` is what makes it a real bar rather than an
               accident: an argmax agrees for free when two logits are a
               float apart, so the gate also states how far apart the top two
               logits were.
"""

from __future__ import annotations

import numpy as np
import pytest

__all__ = ["OBS_SCALE", "TOLERANCE", "decisive", "goes_red",
           "min_decision_gap", "observations", "parity", "require_onnx", "run"]

#: The parity bound, in the units `parity` reports. Measured deviations on the
#: fixtures here are 1.6e-7 (SAC, tanh and the normaliser's clip both active)
#: and 0.0 (PPO's clip), so this is roughly sixty times the noise and a
#: hundredth of the fp32 matmul floor on the chip. A structural defect (a
#: transposed weight, a dropped tanh, a skipped rescale, the normaliser
#: missing or applied twice) moves the action by a fraction of its own range,
#: which is four orders of magnitude outside it.
TOLERANCE = 1e-5

#: Per-column scales for the fixture observations: one column near 0.1 and one
#: near 10, which is the spread a running normaliser exists for.
OBS_SCALE = np.array([1.0, 5.0, 0.2])


def require_onnx():
    """The two packages this gate needs, or a skip naming them."""
    onnx = pytest.importorskip("onnx", reason="needs the tt-rl[onnx] extra")
    runtime = pytest.importorskip(
        "onnxruntime", reason="needs onnxruntime, which the dev extra installs")
    return onnx, runtime


def observations(obs_dim: int, *, seed: int = 11, rows: int = 256) -> np.ndarray:
    """Observations spanning the training distribution AND far outside it.

    The second half is 40x out, which saturates the normaliser's clip and
    drives tanh flat. Both halves are needed: a graph missing the clip agrees
    everywhere inside it, and a graph missing tanh agrees nowhere.
    """
    rng = np.random.default_rng(seed)
    near = rng.normal(size=(rows, obs_dim)) * OBS_SCALE[:obs_dim]
    return np.concatenate([near, near * 40.0]).astype(np.float32)


def run(model, obs: np.ndarray) -> np.ndarray:
    """The graph's action for `obs`. `model` is a path or a `ModelProto`."""
    _onnx, runtime = require_onnx()
    payload = model if isinstance(model, str) else model.SerializeToString()
    session = runtime.InferenceSession(
        payload, providers=["CPUExecutionProvider"])
    return session.run(["action"], {"obs": obs})[0]


def parity(policy, model, obs: np.ndarray) -> float:
    """How far the graph's action is from the artifact's, in `TOLERANCE` units."""
    got = run(model, obs)
    want = policy(obs)
    if policy.action_map == "argmax":
        if got.shape != want.shape:
            return 1.0
        return float(np.mean(got != want))
    half_width = (policy.action_high - policy.action_low) * 0.5
    if got.shape != want.shape:
        return 1.0
    return float(np.max(np.abs(got - want) / half_width))


def min_decision_gap(policy, obs: np.ndarray) -> float:
    """The smallest top-two logit gap over `obs`, which is what makes an
    exact argmax comparison meaningful rather than lucky."""
    ordered = np.sort(policy.head(obs), axis=-1)
    return float(np.min(ordered[:, -1] - ordered[:, -2]))


def decisive(policy, obs: np.ndarray, floor: float) -> np.ndarray:
    """Which observations the argmax is not a coin flip on, above `floor`.

    For comparing against the CHIP rather than against another host. A device
    logit sits about 1e-3 relative from the host's, because the device matmul
    accumulates through HiFi4 and numpy's does not, so where two actions are
    within that of each other the two sides may legitimately pick differently
    and an index comparison there measures rounding. A barely-trained policy
    has many such rows, so the mask is reported alongside the comparison
    rather than hidden inside it.
    """
    ordered = np.sort(policy.head(obs), axis=-1)
    return (ordered[:, -1] - ordered[:, -2]) > floor


def goes_red(policy, model, obs: np.ndarray) -> str | None:
    """Why this graph is not the policy, or None when it is.

    A graph that will not load, one the checker rejects and one that loads and
    computes the wrong action are all the same failure from a deployer's seat:
    the file does not deliver the policy. One function reports all three, so a
    fault-injection table has one column and no injection can pass by failing
    in a way the test forgot to look for.
    """
    try:
        error = parity(policy, model, obs)
    except Exception as exc:  # the graph does not run at all
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    if error > TOLERANCE:
        return f"parity {error:.3e} exceeds {TOLERANCE:.0e}"
    return None
