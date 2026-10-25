"""`check_env` is a strict superset of what a run enforces, permanently.

`ttrl.check_env` used to PASS an environment that `ttrl.build_env` then
REFUSED: the checker drove the episode key by stepping, the enforced contract
checked a class attribute without stepping, and a run called only the second.
Users read `check_env` as a gate meaning "this will train", and it was a
strict subset of the gate that decides whether it will.

This pins the relationship by DERIVING it rather than by listing clauses.
`check_env_contract` is the construction-time contract, and what is asserted
is that all three entry points route through that one function: replace it
with a sentinel and every one of them must raise the sentinel. A clause added
inside it is therefore covered by all three without this file being touched,
which is the property a list of clause names cannot have. Seven typed-literal
coverage tests in this repo have gone stale and two of them hid a shipped bug.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttrl
from tests._fakes.train_layer import ToyEnv
from ttrl.envs import check_env, check_env_contract, check_vec_env
from ttrl.envs.registry import ENVS, register
from ttrl.envs.vector import SyncVectorEnv
from ttrl.train.builders import require_episode_bound as run_requires


class Sentinel(Exception):
    """Raised by the stand-in contract. Nothing else in the tree raises it."""


def _refuse(env, *, where="the environment"):
    raise Sentinel(where)


def _refuse_bound(env, *, where):
    raise Sentinel(where)


def _registered(name, factory):
    register(name, factory)
    return name


def test_nothing_a_run_enforces_escapes_the_public_checkers(monkeypatch):
    """Both halves: the routing, then a live clause through every door.

    `check_env_contract` is patched on `ttrl.envs.check`, which is where
    `check_env` looks it up and where `build_env` imports it from, so one
    patch covers every caller that goes through the contract and misses any
    that does not. Then the patch comes off and one real clause is driven
    through all three, because a derivation over a contract that refuses
    nothing would hold vacuously.
    """
    monkeypatch.setattr("ttrl.envs.check.check_env_contract", _refuse)
    with pytest.raises(Sentinel):
        check_env(ToyEnv(), steps=4)
    vec = SyncVectorEnv([ToyEnv] * 2)
    try:
        with pytest.raises(Sentinel):
            check_vec_env(vec, steps=4)
    finally:
        vec.close()
    name = _registered("superset_toy", lambda **kwargs: ToyEnv(**kwargs))
    try:
        with pytest.raises(Sentinel):
            ttrl.build_env(ttrl.EnvConfig(name=name, n_envs=2))
    finally:
        ENVS.unregister(name)

    monkeypatch.undo()

    class NoSpecs(ToyEnv):
        obs_spec = None

    with pytest.raises(TypeError, match="obs_spec"):
        check_env_contract(NoSpecs())
    with pytest.raises(TypeError, match="obs_spec"):
        check_env(NoSpecs(), steps=4)
    name = _registered("nospec_toy", lambda **kwargs: NoSpecs(**kwargs))
    try:
        with pytest.raises(TypeError, match="obs_spec"):
            ttrl.build_env(ttrl.EnvConfig(name=name, n_envs=2))
    finally:
        ENVS.unregister(name)


def test_a_bare_environment_passes_the_checker_and_then_builds():
    """The divergence itself, as a user met it: check, then train.

    `ToyEnv` wraps nothing. Both have to accept it, and a build has to yield
    an env that reports the score a run reads out of `info`.
    """
    check_env(ToyEnv(), steps=8)
    name = _registered("checked_toy", lambda **kwargs: ToyEnv(**kwargs))
    try:
        env = ttrl.build_env(ttrl.EnvConfig(name=name, n_envs=2))
    finally:
        ENVS.unregister(name)
    try:
        check_vec_env(env, steps=40)
        env.reset(seed=0)
        env.step(np.zeros(env.num_envs, dtype=np.int64))
    finally:
        env.close()


class Endless(ToyEnv):
    """A `ToyEnv` that declares no bound, which is what a run refuses."""

    episode_bound = None


def test_the_episode_bound_a_run_requires_is_asked_by_check_env(monkeypatch):
    """`check_env` used to PASS an environment `Trainer.__init__` refused.

    A toy poker env that ends after four hands passed `ttrl.check_env` and
    was then refused by `ttrl.run`, with a message naming a wrapper rather
    than the one line its author needed. Derived the same way as the contract
    above: `ttrl.envs.bounds.require_episode_bound` is the one question and
    the one message, so ONE patch of it has to reach both doors. Then the
    patch comes off and the live refusal is driven through both.
    """

    monkeypatch.setattr("ttrl.envs.bounds.require_episode_bound", _refuse_bound)
    name = _registered("unbounded_toy", lambda **kwargs: Endless(**kwargs))
    try:
        with pytest.raises(Sentinel):
            check_env(Endless(), steps=4)
        with pytest.raises(Sentinel):
            run_requires(ttrl.EnvConfig(name=name, n_envs=2), where="eval_env")

        monkeypatch.undo()

        for raises in (lambda: check_env(Endless(), steps=4),
                       lambda: run_requires(ttrl.EnvConfig(name=name, n_envs=2),
                                            where="eval_env")):
            with pytest.raises(ValueError) as refusal:
                raises()
            # The author's own one-line fix, named before the wrapper.
            text = str(refusal.value)
            assert text.index("episode_bound = ") < text.index("TimeLimit")
            assert "Endless" in text
        # The declaration an endless environment makes instead.
        check_env(Endless(), steps=4, bounded=False)
    finally:
        ENVS.unregister(name)
