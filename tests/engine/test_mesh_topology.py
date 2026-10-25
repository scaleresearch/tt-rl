"""Chip resolution and physical-to-local id mapping."""

from __future__ import annotations

import pytest

from ttrl.train import replicas
from ttrl.engine.device.scope import ChipScopeError
from ttrl.engine.mesh.topology import (
    ParamSyncError,
    check_visible_scope as _check_visible_scope,
    local_ids_for as _local_ids_for,
    resolve_chips,
)
from ttrl.train.replicas import resolve_run_chips


def test_chips_all_takes_whatever_was_discovered():
    assert resolve_chips("all", [2, 3, 0, 1]) == [0, 1, 2, 3]
    assert resolve_chips(3, [0, 1, 2, 3]) == [0, 1, 2]

def test_chips_all_refuses_an_empty_discovery_with_an_action():
    with pytest.raises(ParamSyncError, match="TT_VISIBLE_DEVICES"):
        resolve_chips("all", [])


def test_chips_explicit_list_validated_against_discovered():
    assert resolve_chips([0, 1], [0, 1, 2, 3]) == [0, 1]
    with pytest.raises(ParamSyncError, match="not among the discovered"):
        resolve_chips([0, 5], [0, 1, 2, 3])


def test_visible_scope_accepts_a_job_that_uses_every_visible_chip():
    _check_visible_scope([0, 1], usable=[0, 1])
    _check_visible_scope([2, 3], usable=[3, 2])  # order is irrelevant
    _check_visible_scope([0, 1, 2, 3], usable=[0, 1, 2, 3])

def test_visible_scope_rejects_a_wider_visible_set_and_names_the_env_var():
    with pytest.raises(ParamSyncError) as excinfo:
        _check_visible_scope([0, 1], usable=[0, 1, 2, 3])
    message = str(excinfo.value)
    assert "TT_VISIBLE_DEVICES=0,1" in message
    assert "[2, 3]" in message  # names the chips that would be starved

def test_visible_scope_rejects_chips_this_process_cannot_see():
    with pytest.raises(ParamSyncError, match=r"not usable by this process"):
        _check_visible_scope([2, 3], usable=[0, 1])


def test_local_ids_never_alias_two_replicas_onto_one_chip(monkeypatch):
    # The exact case `ttrl.engine.device.scope.local_index`'s "prefer the local
    # reading when in range" heuristic gets wrong: visible [1, 2] with chips
    # [1, 2] maps both to index 1 under that heuristic.
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "1,2")
    assert _local_ids_for([1, 2]) == [0, 1]

def test_local_ids_reject_a_chip_outside_the_visible_set(monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ChipScopeError):
        _local_ids_for([0, 1])

def test_chips_all_is_resolved_against_the_visible_set(monkeypatch):
    """`chips: all` must never resolve to chips this process cannot open."""

    monkeypatch.setattr(replicas, "physical_chips", lambda: [0, 1, 2, 3])

    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    assert resolve_run_chips("all") == [0, 1, 2, 3]

    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    assert resolve_run_chips("all") == [2, 3]
    assert resolve_run_chips([2, 3]) == [2, 3]
    with pytest.raises(ParamSyncError, match="not among the discovered"):
        resolve_run_chips([0, 1])
