"""ttrl -- reinforcement learning on Tenstorrent, natively on TT-NN."""

import importlib


__version__ = "0.1.0"

# Do not eagerly import TT-NN. Besides keeping CPU-only imports possible, this
# is a correctness requirement for chip-scoped subprocesses: environment scope
# must exist before the first ``import ttnn``.
__all__ = [
    "ObsSpec",
    "ObsField",
    "ObsGroup",
    "ObsLeaf",
    "Observation",
    "DEFAULT_LEAF",
    "ActionSpec",
    "mask_rows",
    "group_mask_rows",
    "draw_legal",
    "check_legal",
    "Agent",
    "AlgoConfig",
    "check_algo_config",
    "divide_for_replicas",
    "register_checkpoint_type",
    "Transition",
    "check_agent",
    "ALGOS",
    "ENCODERS",
    "ENVS",
    "SAMPLERS",
    "ReplaySampler",
    "check_sampler",
    "UpdateCadence",
    "StepSource",
    "check_cadence",
    "EpisodeBoundaries",
    "check_boundaries",
    "check_env",
    "check_env_contract",
    "check_vec_env",
    "build_env",
    "RecordEpisodeStatistics",
    "DQNConfig",
    "PPOConfig",
    "RecurrentConfig",
    "PriorityConfig",
    "SACConfig",
    "TrainConfig",
    "DeviceConfig",
    "EnvConfig",
    "load_config",
    "device_for",
    "Trainer",
    "run",
    "evaluate",
    "evaluate_checkpoint",
    "record_video",
    "config_from_checkpoint",
    "agent_from_checkpoint",
    "export_policy",
    "export_onnx",
    "load_policy",
    "Logger",
    "NullLogger",
    "JsonlLogger",
    "StdoutLogger",
    "MultiLogger",
    "build_logger",
    "TensorBoardLogger",
    "WandbLogger",
    "TrainHooks",
    "RunState",
    "StopOnRewardThreshold",
    "StopOnNoImprovement",
    "AnyOf",
    "algos",
    "deploy",
    "engine",
    "envs",
    "nets",
    "spaces",
    "train",
    "__version__",
]

#: Subpackages reachable as `ttrl.<name>` without importing them first. Each
#: is imported on the first attribute access, which is what keeps `import
#: ttrl` free of ttnn and of every environment suite.
#: `buffers` is deliberately absent: an agent owns its own experience storage
#: (see `ttrl.algos.base`), no contract asks an application author to touch a
#: `ReplayBuffer`, and a public name over an implementation detail is a
#: promise about its shape that nothing here intends to keep. It stays
#: importable as `ttrl.buffers` for the agents that use it.
_SUBPACKAGES = ("algos", "deploy", "engine", "envs", "nets", "spaces", "train")

#: Names reached through `ttrl.<name>`, mapped to the module that owns each.
#: Resolution is deferred so that `import ttrl` imports neither ttnn nor any
#: environment suite.
_HOMES = {
    # An observation is an `ObsGroup` of leaves and an `Observation` is what
    # crosses every boundary one crosses, the exported `Agent.observe`
    # included, so an author writing an environment or an agent names all
    # four from here. `DEFAULT_LEAF` is the leaf name a single-array
    # observation carries, which is what such an author reads an
    # `Observation`, a batch and a deployed artifact's parameter names by.
    #
    # The four mask names are here because action masking is one of this
    # library's two strongest features and was invisible from the public
    # surface: a reader of `ttrl.__all__` alone concluded that masking is
    # unsupported. An author writes all four. `mask_rows` reads the tail of
    # one leaf's observations, `group_mask_rows` does it for a whole
    # `ObsGroup` and is what an author with more than one leaf needs,
    # `draw_legal` is how a random or epsilon action is drawn legally, and
    # `check_legal` is how an action that reached storage from outside a
    # masking `act` is caught.
    #
    # `require_group` stays a `ttrl.spaces` name: the library calls it on its
    # own arguments and an application author never does. Same for the kind
    # aliases.
    "ttrl.spaces": ("ObsSpec", "ObsField", "ObsGroup", "ObsLeaf",
                    "Observation", "DEFAULT_LEAF", "ActionSpec",
                    "check_legal", "draw_legal", "group_mask_rows",
                    "mask_rows"),
    # `Transition` is the type of the sole argument to `Agent.observe`, and
    # `check_agent` is how a third-party agent is held to the protocol, so
    # both belong wherever `Agent` is reachable from.
    # No `build_agent`. Of the two functions with that name, the one here
    # resolves a registered algorithm and constructs it against a live device
    # and live specs WITHOUT running `check_agent`, and the one in
    # `ttrl.train.builders` is config-driven and does check. Exporting the
    # unchecked one hands an author the builder whose broken contract surfaces
    # at the first `observe`, which is the failure `build_env` was just made to
    # refuse. An author writes a `TrainConfig` and calls `ttrl.run`; someone
    # constructing an agent directly reaches the resolver through `ALGOS`,
    # which is the right depth for that need.
    "ttrl.algos": ("Agent", "ALGOS", "Transition", "check_agent"),
    # What the object in `TrainConfig.algo_cfg` must provide. A third-party
    # algorithm's config is held to it by `check_algo_config`, which
    # `TrainConfig` runs on whatever it is handed.
    "ttrl.algos.config": ("AlgoConfig", "check_algo_config"),
    # The one piece of `scale_for_replicas` nobody should write twice. The
    # contract names it, so it is reachable from where the contract is.
    "ttrl.algos.scaling": ("divide_for_replicas",),
    # How a config type an outside package owns gets into a checkpoint and
    # back out. The owning package registers it; the file never names a type
    # the reading process has not already imported.
    "ttrl.train.checkpoint.types": ("register_checkpoint_type",),
    # The architecture seam, and the two checkers that turn the environment
    # contract into an executable test for whoever writes an adapter.
    "ttrl.nets": ("ENCODERS",),
    # `check_env_contract` is the construction-time subset that a run itself
    # enforces, which is why it is exported beside the two drivers: a caller
    # holding an env it did not build can ask for exactly what `build_env`
    # will ask.
    "ttrl.envs": ("ENVS", "check_env", "check_env_contract", "check_vec_env"),
    # Episode statistics are the vector env's own work, so this wrapper is no
    # longer required of anyone. It is exported for the case the vector level
    # cannot serve: a lane that transforms its own reward takes the return
    # BELOW that transform, and only a wrapper inside the lane sits there.
    "ttrl.envs.wrappers": ("RecordEpisodeStatistics",),
    # An algorithm's config lives with the algorithm, so these resolve from
    # ttrl.algos rather than from ttrl.train.config. Each subpackage imports
    # its config eagerly and its agent lazily, so resolving a config here
    # stays free of ttnn too, which is what lets a config file be read and
    # validated on a machine with no device.
    # The one `ttrl.buffers` name an application author writes, and the
    # exception the paragraph above anticipates: `DQNConfig.priority` and
    # `SACConfig.priority` take one of these, so a config file cannot be
    # written without naming the type. The buffer classes stay unexported.
    "ttrl.buffers.sampling": ("PriorityConfig",),
    # The host-side learning seams, and the checkers that turn each into an
    # executable test. A sampler is the one piece of `ttrl.buffers` a user
    # replaces, so its registry, its protocol and its checker are named here
    # even though the buffer classes are not: a seam a user cannot name from
    # the public surface is not declared.
    "ttrl.buffers.samplers": ("SAMPLERS", "ReplaySampler"),
    "ttrl.buffers.checks": ("check_sampler",),
    # `UpdateCadence` and `StepSource` document what the loop already calls,
    # so that replacing either is a thing the surface says is allowed.
    "ttrl.train.cadence": ("UpdateCadence", "check_cadence"),
    "ttrl.train.loop": ("StepSource",),
    # Where a recurrent policy's memory resets. The flags were PPO's own
    # private business, which left the first question a recurrent policy
    # invites -- is it using its memory at all? -- reachable only through two
    # private attributes, one of them on an unexported buffer class.
    "ttrl.algos.ppo.boundaries": ("EpisodeBoundaries", "check_boundaries"),
    "ttrl.algos.dqn": ("DQNConfig",),
    "ttrl.algos.ppo": ("PPOConfig", "RecurrentConfig"),
    "ttrl.algos.sac": ("SACConfig",),
    # The config dataclasses are exported because writing a run *is* writing
    # one: `configs/*.py` build a `TrainConfig(...)` literal out of them.
    "ttrl.train.config": ("TrainConfig", "DeviceConfig", "EnvConfig", "load_config"),
    # The one public path to an open chip. `ttrl.engine.acquire` needs the
    # process scoped before ttnn loads and cannot say so in its signature;
    # `device_for` does both in the only order that works.
    "ttrl.train.device": ("device_for",),
    # The one public path from an `EnvConfig` to a stepping vector env. There
    # was none, which made "build the env from the config and step it" a
    # thing the public surface could not express at all, including in this
    # library's own release criteria. It applies the normalisers the config
    # asks for and runs `check_env_contract`, so it is the same env a run
    # trains on rather than a hand-assembled lookalike.
    "ttrl.train.builders": ("build_env",),
    "ttrl.train.trainer": ("Trainer",),
    "ttrl.train.run": ("run",),
    "ttrl.train.evaluate": ("evaluate", "evaluate_checkpoint"),
    # Watching a policy is a first-class use of a library aimed at games and
    # locomotion, and doing it meant copying an example that hardcoded one
    # environment builder. One lane, through the `Renderable` protocol.
    "ttrl.train.record": ("record_video",),
    "ttrl.train.restore": ("config_from_checkpoint", "agent_from_checkpoint"),
    # Leaving a run: a checkpoint becomes a numpy-only artifact, and that
    # artifact is read back. The deployed side imports numpy alone, so a
    # robot host reaches `load_policy` without the training stack.
    # `export_onnx` turns that artifact into an ONNX graph, which is what the
    # robotics deployment ecosystem reads: an optional `onnx` extra, refused
    # by name when it is absent.
    "ttrl.deploy": ("export_policy", "export_onnx", "load_policy"),
    # A run takes a `Logger`, so it has to be possible to get one from here.
    # Exporting the protocol without an implementation of it left every
    # caller writing its own JSONL sink.
    "ttrl.train.logging": (
        "Logger",
        "NullLogger",
        "JsonlLogger",
        "StdoutLogger",
        "MultiLogger",
        "build_logger",
        "TensorBoardLogger",
        "WandbLogger",
    ),
    # A run takes hooks the same way it takes a logger, so the same argument
    # applies: the ready-made stop conditions and the `RunState` a condition
    # of one's own reads are exported, while `BestScore` is not. That one is
    # the Trainer's own bookkeeping, constructed by the Trainer and never by
    # an application author, and a run reports what it decided through
    # `result["best"]`.
    "ttrl.train.hooks": ("TrainHooks",),
    "ttrl.train.stop": (
        "AnyOf",
        "RunState",
        "StopOnNoImprovement",
        "StopOnRewardThreshold",
    ),
}

_OWNER = {name: module for module, names in _HOMES.items() for name in names}


def __getattr__(name):
    # NOTE: `ttrl.train` names the `train/` subpackage, so the one-call entry
    # point is exported as `ttrl.run`: a submodule import always wins its own
    # name over anything `__getattr__` returns.

    if name in _SUBPACKAGES:
        return importlib.import_module(f"ttrl.{name}")
    module = _OWNER.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    # Without this, lazily-exported names are invisible to `dir(ttrl)` and to
    # REPL/IDE completion, since none of them exist in the module dict yet.
    return sorted(set(__all__) | set(globals()))
