"""The teardown ladder: every step runs, the first failure wins, nothing masks."""

from __future__ import annotations

import warnings

import pytest

from ttrl.train.teardown import Step, run_ladder


def _step(label, call):
    return Step(label, "do the thing the label implies.", call)


def test_every_step_runs_even_after_one_raises():
    ran = []

    def boom():
        ran.append("boom")
        raise OSError("env worker")

    with pytest.raises(OSError, match="env worker"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_ladder([
                _step("a", lambda: ran.append("a")),
                _step("b", boom),
                _step("c", lambda: ran.append("c")),
            ])
    assert ran == ["a", "boom", "c"]


def test_the_first_failure_is_the_one_raised():
    def first():
        raise OSError("first")

    def second():
        raise RuntimeError("second")

    with pytest.warns(UserWarning, match="was not the first failure"):
        with pytest.raises(OSError, match="first"):
            run_ladder([_step("a", first), _step("b", second)])


def test_a_teardown_error_never_masks_the_real_failure():
    """The whole point: the exception that ended the run is what propagates."""

    def closing():
        run_ladder([_step("env", _raise(OSError("flaky close")))])

    with pytest.warns(UserWarning, match="already propagating"):
        with pytest.raises(ZeroDivisionError, match="the real failure"):
            try:
                raise ZeroDivisionError("the real failure")
            finally:
                closing()


def _raise(exc):
    def call():
        raise exc

    return call
