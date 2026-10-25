"""The one fixture the train-layer tests share."""

from __future__ import annotations

import pytest

from tests._fakes.policy_state import RecordingAgent
from ttrl.train import builders


@pytest.fixture
def agents(monkeypatch):
    """Every agent the trainer builds, as `RecordingAgent`s."""
    built = []

    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None,
                    algo_cfg=None):
        agent = RecordingAgent(obs_spec, action_spec)
        built.append(agent)
        return agent

    monkeypatch.setattr(builders, "build_agent", build_agent)
    return built
