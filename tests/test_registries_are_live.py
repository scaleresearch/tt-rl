"""No shipped registry accepts a registration that does nothing.

A `Registry` is a promise: register a builder, name it in a config, and the
library uses it. `NETS` broke that promise for as long as it existed. Nothing
called `build_net`, no config field named a net, and `@NETS("my_net")`
therefore registered into a dictionary that was read by nothing, with no
failure anywhere to say so.

This test is the guard. Every `Registry` built in `ttrl/` has to appear in
`CONSUMERS` below with the config field a name travels in and the call that
resolves it, and both ends are then checked. A new registry with no config
path fails here, and so does a registry whose consumer stops resolving it.

The source is read as text rather than imported: `ttrl.nets.encoders` and the
three agents reach ttnn, and this question is answerable without a chip.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

import pytest

from tests._fakes.train_layer import make_toy
import ttrl
import ttrl.envs
from ttrl.envs import ENVS
import ttrl.nets
from ttrl.nets import (
    build_actor_critic,
    build_q_net,
    build_recurrent_actor_critic,
    build_sac_nets,
)
from ttrl.nets.choice import UnnamedEncoder, resolve_encoder
from ttrl.train.builders import build_env
from ttrl.train.checkpoint.tree import join, split
from ttrl.train.config import EnvConfig
from ttrl.train.provenance import run_metadata

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "ttrl"

#: registry name -> (file that builds it, config fields naming an entry)
CONSUMERS = {
    "ENCODERS": (
        "ttrl/nets/registry.py",
        ("ttrl/algos/dqn/config.py:encoder", "ttrl/algos/ppo/config.py:encoder",
         "ttrl/algos/sac/config.py:encoder"),
    ),
    "ALGOS": (
        "ttrl/algos/registry.py",
        ("ttrl/train/config/schema.py:algo",),
    ),
    "ENVS": (
        "ttrl/envs/registry.py",
        ("ttrl/train/config/env.py:name",),
    ),
    "SAMPLERS": (
        "ttrl/buffers/samplers.py",
        ("ttrl/algos/dqn/config.py:sampler", "ttrl/algos/sac/config.py:sampler"),
    ),
}

_BUILT = re.compile(r"^(\w+) = Registry\(", re.MULTILINE)


def _registry_sites() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in _BUILT.findall(path.read_text()):
            found[name] = str(path.relative_to(REPO))
    return found


def test_every_registry_in_the_package_is_listed_here():
    """A registry added without a config path has no way to be reached."""
    sites = _registry_sites()
    assert set(sites) == set(CONSUMERS), (
        f"registries built in ttrl/ are {sites}, and the ones with a declared "
        f"config path are {sorted(CONSUMERS)}. A registry nothing resolves "
        f"from a config accepts registrations that do nothing; give it a "
        f"config field, or delete it and say in its module what the real "
        f"extension seam is."
    )
    for name, (where, _fields) in CONSUMERS.items():
        assert sites[name] == where


@pytest.mark.parametrize("name", sorted(CONSUMERS))
def test_a_config_field_carries_a_name_into_the_registry(name):
    _where, fields = CONSUMERS[name]
    for field in fields:
        path, attribute = field.split(":")
        source = (REPO / path).read_text()
        assert re.search(rf"^\s+{attribute}: ", source, re.MULTILINE), (
            f"{path} declares no {attribute!r} field, so nothing a user writes "
            f"in a config can name an entry of {name}."
        )
    # Searched across the package rather than at one path: which module holds
    # the call is free to move, that something calls it is not.
    callers = [str(path.relative_to(REPO)) for path in PACKAGE.rglob("*.py")
               if f"{name}.resolve(" in path.read_text()]
    assert callers, (
        f"nothing in ttrl/ calls {name}.resolve(), so a registration into "
        f"{name} is read by nothing."
    )


def test_the_encoder_field_reaches_the_encoder_builder():
    """The field is threaded through, not merely declared."""
    for agent in ("dqn", "ppo", "sac"):
        # Searched across the algorithm's package, for the reason above: which
        # module holds the call is free to move, that something makes it is not.
        package = PACKAGE / "algos" / agent
        passes = [path.name for path in package.rglob("*.py")
                  if "encoder=cfg.encoder" in path.read_text()]
        assert passes, (
            f"nothing in ttrl/algos/{agent}/ passes cfg.encoder to a net "
            f"builder, so the encoder named in its config is ignored."
        )
        assert any("cfg.encoder_kwargs" in path.read_text()
                   for path in package.rglob("*.py")), (
            f"nothing in ttrl/algos/{agent}/ passes cfg.encoder_kwargs to a "
            f"net builder, so an encoder width needs a registration per width."
        )
    # Every net builder an algorithm may call, including the recurrent one,
    # takes an encoder name and hands it to build_encoder. Checked by
    # signature as well as by source, because a builder that quietly dropped
    # the parameter would leave the string above passing while the tower it
    # built came from a default.

    builders = (build_actor_critic, build_q_net, build_recurrent_actor_critic,
                build_sac_nets)
    for builder in builders:
        assert "encoder" in inspect.signature(builder).parameters, (
            f"{builder.__name__} takes no encoder, so an algorithm calling it "
            f"cannot thread the name its config declares."
        )
    threaded = sum(
        (PACKAGE / "nets" / module).read_text().count("name=encoder")
        for module in ("build.py", "recurrent.py")
    )
    assert threaded >= len(builders), (
        f"ttrl/nets passes name=encoder to build_encoder {threaded} times for "
        f"{len(builders)} builders; one of them resolves its tower from "
        f"something other than the config's encoder name."
    )


@pytest.fixture
def registered_env():
    """A name this module registers itself, and takes back out afterwards.

    The toy builder used to be reached by the registration `tests._fakes`
    performs when imported, so this test passed in a full suite and failed on
    its own. A fixture names what it needs instead.
    """

    name = "registries-are-live-toy"
    ENVS.register(name, make_toy)
    try:
        yield name
    finally:
        ENVS.unregister(name)


def test_an_env_name_in_a_config_is_resolved_by_the_builder(registered_env):
    """The env end is checked by running it: the registry is host-only."""

    env = build_env(EnvConfig(name=registered_env, n_envs=2))
    try:
        assert env.num_envs == 2
        assert env.obs_spec is not None and env.action_spec is not None
    finally:
        env.close()


def test_the_three_registries_share_one_set_of_verbs():
    """A user who registered an encoder guesses the env spelling and is right."""
    # Imported here: other tests purge ttrl from sys.modules, which would leave
    # a module-level binding pointing at a stale registry.
    from ttrl.buffers.samplers import SAMPLERS

    # All four hold target strings, so only `resolve` imports a builder, and
    # with it ttnn; only the verbs themselves are asked for here.
    for registry in (ttrl.nets.ENCODERS, ttrl.envs.ENVS, ttrl.SAMPLERS):
        for verb in ("register", "resolve", "names", "unregister", "entry"):
            assert callable(getattr(registry, verb)), (verb, registry)
    assert "definitely-not-registered" not in ttrl.envs.ENVS
    assert ttrl.envs.register == ttrl.envs.ENVS.register
    # The three are reachable by the same spelling from the top level.
    assert ttrl.ENVS is ttrl.envs.ENVS and ttrl.ENCODERS is ttrl.nets.ENCODERS
    assert ttrl.SAMPLERS is SAMPLERS


#: What writing an encoder out of layers takes. Without these on `ttrl.nets`
#: the author names `ttrl.engine.layers`, which nothing advertises, and has no
#: `Module` to subclass: an object that is not one is skipped by discovery.
ENCODER_AUTHORING_NAMES = ("Module", "Linear", "LayerNorm", "ReLU", "Tanh",
                           "Sequential", "Flatten", "Conv2d", "Encoder", "ScaledLinear")


@pytest.mark.parametrize("name", ENCODER_AUTHORING_NAMES)
def test_an_encoder_of_ones_own_can_be_written_from_the_nets_surface(name):
    """Advertised, and backed by a definition: by name rather than by
    resolution, for the reason at the top of this file."""
    assert name in ttrl.nets.__all__ and name in dir(ttrl.nets), (
        f"ttrl.nets does not advertise {name!r}, so an encoder of one's own "
        f"has to name a module nothing advertises."
    )
    assert [p for p in PACKAGE.rglob("*.py")
            if re.search(rf"^class {name}\b", p.read_text(), re.M)], (
        f"ttrl.nets advertises {name!r} and nothing defines it")


class MyTower:
    """Stands in for the architecture a user tries before naming it."""


def _round_trip(config):
    """`config` through the checkpoint writer and reader, as a file would."""

    arrays: dict = {}
    node = split(config, "config", arrays)
    return join(node, arrays, file="<memory>")


#: Every exported config that names an encoder, which is what this gate is
#: about. Derived from the field rather than from a list of the configs that
#: do not: a new algorithm config is covered the moment it is exported, and a
#: config with no encoder to hand over is not listed here to be excluded.
def _encoder_bearing_configs():
    found = []
    for name in ttrl.__all__:
        obj = getattr(ttrl, name)
        if (dataclasses.is_dataclass(obj) and isinstance(obj, type)
                and any(f.name == "encoder" for f in dataclasses.fields(obj))):
            found.append(obj)
    assert found, "no exported config declares an encoder"
    return sorted(found, key=lambda cls: cls.__name__)


@pytest.mark.parametrize("config_cls", _encoder_bearing_configs())
def test_a_handed_over_encoder_class_survives_a_config_and_says_what_it_cannot(
        config_cls):
    """The class is taken, and the checkpoint is honest about it.

    Every algorithm config is covered, derived from `ttrl.__all__`: one that
    took a name only would fail here rather than be forgotten. A NAME still
    round-trips through a file and still resolves on its own; a CLASS comes
    back as the identity it was with no builder, and asking it to build
    refuses with the two fixes named.
    """

    named = _round_trip(config_cls(encoder="mlp"))
    assert named.encoder == "mlp"

    handed = config_cls(encoder=MyTower)
    assert isinstance(handed.encoder, UnnamedEncoder)
    assert handed.encoder.resolve() is MyTower
    assert handed.encoder.target == f"{__name__}:MyTower"

    restored = _round_trip(handed)
    # Same identity, so re-supplying the class resumes the run rather than
    # reporting a changed architecture.
    assert restored.encoder == handed.encoder
    assert restored.encoder.builder is None
    with pytest.raises(ValueError, match=r"ENCODERS.register"):
        restored.encoder.resolve()

    # The field reaches a tower the same way a name does, alone and as one
    # leaf of a mapping. `resolve_encoder` is what `build_encoder` calls; the
    # `default` is never resolved here, so this costs no ttnn import.
    leaves = ("pixels", "proprio")
    assert resolve_encoder(handed.encoder, "pixels", leaves,
                           default="unused") is MyTower
    mixed = config_cls(encoder={"pixels": MyTower, "proprio": "mlp"})
    assert resolve_encoder(mixed.encoder, "pixels", leaves,
                           default="unused") is MyTower
    with pytest.raises(ValueError, match="no encoder for leaf 'proprio'"):
        resolve_encoder({"pixels": MyTower}, "proprio", leaves,
                        default="unused")

    # A new encoder width is a config value rather than a new registration.
    assert config_cls(encoder_kwargs={"hidden": 96}).encoder_kwargs == {"hidden": 96}
    with pytest.raises(TypeError, match="must be a dict"):
        config_cls(encoder_kwargs=[("hidden", 96)])

    # And the run's own record. `run_metadata` json.dumps what it returns, so
    # a callable left in the config dict killed every run that handed one
    # over, before the chip. The file says the qualified name under a key of
    # its own: a record must not read as though a registered name was used.

    algo = config_cls.__name__.removesuffix("Config").lower()
    record = run_metadata(ttrl.TrainConfig(algo=algo, total_frames=100,
                                           algo_cfg=handed))
    assert json.loads(json.dumps(record))["config"]["algo_cfg"]["encoder"] == {
        "unnamed_encoder": f"{__name__}:MyTower"
    }
