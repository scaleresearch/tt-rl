"""What `ttrl.run` records before it opens anything, and what it refuses there."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess

import pytest

from tests._fakes.train_layer import RecordingLogger, ScriptedAgent
import ttrl.engine
from ttrl.engine.device.scope import ChipScopeError
from ttrl.train import builders, run as run_mod
from ttrl.train.config import DeviceConfig, EnvConfig, TrainConfig
from ttrl.train.gitstate import git_state
from ttrl.train.provenance import COMMIT_ENV_VAR, METADATA_NAME, code_line, git_state
from ttrl.train.trainer import Trainer


def config(tmp_path, **kwargs):
    base = dict(
        total_frames=6,
        eval_interval=1000,
        checkpoint_interval=1000,
        log_dir=str(tmp_path / "run"),
        env=EnvConfig(name="toy", kwargs={"episode_len": 3}),
    )
    return TrainConfig(**{**base, **kwargs})


@pytest.fixture
def agents(monkeypatch):
    def build_agent(cfg, *, obs_spec, action_spec, device, profile=None, algo_cfg=None):
        return ScriptedAgent(obs_spec, action_spec)

    monkeypatch.setattr(builders, "build_agent", build_agent)


@pytest.fixture
def fake_chip(monkeypatch, tmp_path):
    """A one-chip acquire that records the order things happened in."""
    events = []
    record_path = tmp_path / "run" / METADATA_NAME

    @contextlib.contextmanager
    def acquire(chip_id=None, *, cfg=None, **kwargs):
        events.append("record_exists" if record_path.exists() else "no_record")
        events.append("acquire")
        yield ("device", "profile")

    monkeypatch.setattr(run_mod, "acquire", acquire)
    monkeypatch.setattr(run_mod, "scope_for_run",
                        lambda dc: events.append("scope") or 2)
    return events


def _record(cfg):
    with open(os.path.join(cfg.log_dir, METADATA_NAME)) as fh:
        return json.load(fh)


def test_the_record_is_written_before_the_device_is_opened(tmp_path, agents,
                                                           fake_chip):
    """A run killed waiting for a contended chip still leaves a record. With
    `acquire_timeout` settable that wait can be half an hour."""
    run_mod.run(config(tmp_path), logger=RecordingLogger())
    assert fake_chip == ["scope", "record_exists", "acquire"]


def test_the_record_survives_a_scoping_failure(tmp_path, agents, monkeypatch):
    """The reason the record is written BEFORE `scope_for_run` and not just     before `acquire`."""

    def boom(device_config):
        raise ChipScopeError("could not locate the mesh graph descriptor")

    monkeypatch.setattr(run_mod, "scope_for_run", boom)
    cfg = config(tmp_path)
    with pytest.raises(ChipScopeError):
        run_mod.run(cfg, logger=RecordingLogger())

    record = _record(cfg)
    assert record["config"]["total_frames"] == 6
    assert record["resolved_chip"] is None, "scoping never got that far"


def test_the_record_carries_the_resolved_chip_beside_the_request(tmp_path,
                                                                agents,
                                                                fake_chip,
                                                                monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2")
    cfg = config(tmp_path, device=DeviceConfig(chip=None))
    run_mod.run(cfg, logger=RecordingLogger())
    record = _record(cfg)
    assert record["config"]["device"]["chip"] is None, "the request, unchanged"
    assert record["resolved_chip"] == 2, "and the chip actually scoped to"


def test_a_trainer_built_directly_records_its_config(tmp_path, agents,
                                                     monkeypatch):
    """`Trainer` is the path a library user and an outside package compose a
    run through, and a run that records nothing cannot be verified later."""

    monkeypatch.setenv("TT_VISIBLE_DEVICES", "3")
    cfg = config(tmp_path)
    with Trainer(cfg, object(), object(), logger=RecordingLogger()) as trainer:
        trainer.run()
    record = _record(cfg)
    assert record["config"]["total_frames"] == 6
    assert record["resolved_chip"] == 3
    assert record["code"]


def test_an_unregistered_algo_is_refused_before_any_acquire(tmp_path, agents,
                                                            fake_chip):
    cfg = config(tmp_path, algo="rainbow")
    with pytest.raises(ValueError) as excinfo:
        run_mod.run(cfg, logger=RecordingLogger())
    message = str(excinfo.value)
    assert "rainbow" in message and "dqn" in message
    assert fake_chip == ["scope"]



def test_an_exported_tree_records_the_commit_its_caller_states(tmp_path,
                                                               monkeypatch):
    """`git archive` leaves no .git, which is the case provenance is for."""

    sha = "c5056ae" + "0" * 33
    monkeypatch.setenv(COMMIT_ENV_VAR, sha.upper())
    export = tmp_path / "export"
    export.mkdir()
    state = git_state(root=export)
    assert state["head"] == sha
    assert state["source"] == "asserted", "a claim, never read from a repo"
    assert state["asserted_by"] == COMMIT_ENV_VAR
    assert state["dirty"] is None, "nothing here saw the tree's files"
    assert "asserted" in code_line(state, "0")

    assert git_state()["source"] == "checkout", "a real checkout answers itself"

    monkeypatch.delenv(COMMIT_ENV_VAR)
    assert git_state(root=export) == {
        "available": False, "source": "none", "error": "not a git repository"}


def test_a_commit_that_cannot_be_one_is_refused(monkeypatch):
    for bad in ("HEAD", "c5056a", "z" * 40, "c5056ae" + "0" * 40):
        monkeypatch.setenv(COMMIT_ENV_VAR, bad)
        with pytest.raises(ValueError, match="git rev-parse"):
            git_state()


def test_the_dirty_list_names_the_files_that_are_dirty(tmp_path):
    """Four statuses in one real repository, including the one that used to
    lose a character: the status field's first column is a space for a
    worktree-only change, and stripping the output ate it."""

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(("git", "-C", str(repo)) + args, check=True,
                       capture_output=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    (repo / "a modified.py").write_text("one\n")
    (repo / "old_name.py").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    (repo / "a modified.py").write_text("one\ntwo\n")
    (repo / "untracked é.py").write_text("u\n")
    (repo / "staged.py").write_text("z\n")
    git("add", "staged.py")
    git("mv", "old_name.py", "new_name.py")

    state = git_state(root=repo)
    assert state["dirty_file_count"] == 4
    assert state["dirty_files"] == [
        "a modified.py",      # worktree only: status " M"
        "new_name.py",        # the name the tree holds, never "old_name.py"
        "staged.py",
        "untracked é.py",  # quoted and escaped without -z
    ]
