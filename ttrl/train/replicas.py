"""Opening the devices a run needs and building one replica per device.

Construction goes through the `builders` module by attribute rather than by a
bound import, which is what lets a host-only test substitute one constructor
and drive everything else for real.

`config.device.chips == 1` means the caller opened the device and passes it
in, so the Trainer opens nothing and releases nothing. Anything else means
the Trainer opens one INDEPENDENT single-chip device per resolved chip and
owns their lifetime. Independent devices, never a shared mesh: a mesh and
independently-opened devices on the same chips is recorded in-tree as fatal
on this hardware, and the host parameter all-reduce exists for exactly that
reason.
"""

from __future__ import annotations

import contextlib

from ttrl.algos.base import check_agent
from ttrl.engine.device import physical_chips, visible_chip_ids
from ttrl.engine.device.profile import probe_profile
from ttrl.train import builders
from ttrl.train.config import TrainConfig
from ttrl.train.loop import Replica

__all__ = ["ReplicaSet", "open_replicas", "resolve_run_chips"]

#: Stride between replicas' seeds. Large and coprime with any plausible lane
#: count, so two replicas never share a stream.
SEED_STRIDE = 1_000_003


class ReplicaSet:
    """The replicas of one run, the devices they sit on, and their averaging.

    `release_devices()` closes only what this object opened, which is nothing
    at `chips == 1`. The agents and envs are closed by the Trainer's teardown
    ladder first, because an agent holds a captured trace that lives on the
    device it is about to release.
    """

    def __init__(self, replicas, devices, param_sync, stack) -> None:
        self.replicas = replicas
        self.devices = devices
        self.param_sync = param_sync
        self._stack = stack

    @property
    def n(self) -> int:
        return len(self.replicas)

    @property
    def lead(self) -> Replica:
        return self.replicas[0]

    def release_devices(self) -> None:
        self._stack.close()


def open_replicas(config: TrainConfig, device, profile, *, cadence=None,
                  source=None, agent=None) -> ReplicaSet:
    """Build the run's replicas. Releases whatever it opened if it cannot finish.

    The unwind is a separate stack from the one the ReplicaSet keeps, because
    the two have different lifetimes: a half-built run closes its envs and
    agents here and now, while a built one hands that job to the Trainer's
    teardown ladder so a single failure cannot strand the rest.

    `cadence(agent)`, `source(env_config, index)` and `agent(built, index)`
    are the injection points for the replaceable stages, called ONCE PER
    REPLICA and here rather than in the caller because this is where a replica's agent and its env
    come into existence: a cadence wraps an agent that does not exist at
    config time, and one instance shared across replicas is the starvation
    `ttrl.train.cadence.check_cadence` exists to catch. Left None, the agent
    is its own cadence and `builders.build_env` is the source, which is the
    identity of today's behaviour.

    `agent` is handed the agent `ALGOS` already built for this replica and
    returns the one the replica will use, which is how a config-driven run
    reaches a constructor argument the config cannot carry -- a
    `ttrl.EpisodeBoundaries` is the case this exists for. It adapts rather
    than constructs so that the registry stays the one path from an algorithm
    name to an agent: a caller wanting a different algorithm registers it and
    names it in the config. Left None, the built agent is the agent.
    """
    device_stack = contextlib.ExitStack()
    partial = contextlib.ExitStack()
    with partial:
        devices, profiles = _devices(config, device, profile, device_stack)
        partial.callback(device_stack.close)
        replicas = []
        # Each replica collects 1/n of the run's frames, so every budget
        # measured in frames is divided before its agent is built. The
        # algorithm declares which of its fields those are; at n=1 this is the
        # identity.
        algo_cfg = _scaled_algo_config(config, len(devices))
        for index, (dev, prof) in enumerate(zip(devices, profiles)):
            env = _source(config, source, index)
            # Registered before the agent is built, not after: a failure in the
            # agent must still close this replica's env workers.
            partial.callback(env.close)
            built = builders.build_agent(
                config,
                obs_spec=env.obs_spec,
                action_spec=env.action_spec,
                device=dev,
                profile=prof,
                algo_cfg=algo_cfg,
            )
            agent_used = _agent(agent, built, index)
            # Registered on what the hook returned, and only once it has
            # returned: an adapter that replaces the built agent owns it from
            # that point, and one that raises leaves nothing to close here but
            # the agent it was handed, which `_agent` closes itself.
            partial.callback(agent_used.close)
            replica_cadence = None if cadence is None else cadence(agent_used)
            # Only if it has one: `UpdateCadence` declares `should_update` and
            # `update` and nothing else, so a cadence holding no resource has
            # nothing to release and must not be required to say so.
            if replica_cadence is not None and hasattr(replica_cadence, "close"):
                partial.callback(replica_cadence.close)
            seed = None if config.env.seed is None else config.env.seed + index * SEED_STRIDE
            replicas.append(Replica(agent_used, env, index=index, seed=seed,
                                    cadence=replica_cadence))
        param_sync = _param_sync(config, replicas)
        partial.pop_all()
    return ReplicaSet(replicas, devices, param_sync, device_stack)


def _agent(agent, built, index: int):
    """This replica's agent: the caller's adapter over the built one, or it.

    Every refusal below closes the agent it was handed, including one raised
    inside the adapter itself: the built agent holds staging on an open chip
    and the caller's unwind stack has not been told about it yet, so this is
    the only code that can still reach it.
    """
    if agent is None:
        return built
    try:
        return _adapted(agent, built, index)
    except BaseException:
        built.close()
        raise


def _adapted(agent, built, index: int):
    """The adapter's answer, held to the `Agent` surface.

    Checked with `ttrl.algos.base.check_agent`, the same function
    `builders.build_agent` runs on what the registry built, so an adapter is
    held to exactly the surface a registered algorithm is and the member list
    has one definition.
    """

    if not callable(agent):
        raise TypeError(
            f"agent= is a {type(agent).__name__} and is not callable. It is a "
            f"factory called once per replica as agent(built, index), where "
            f"`built` is the agent ttrl.ALGOS built for that replica. Pass a "
            f"function, e.g. "
            f"lambda built, index: setattr(built, 'boundaries', Mine()) or built."
        )
    adapted = agent(built, index)
    if adapted is None:
        raise TypeError(
            f"the agent= factory returned None for replica {index}. It returns "
            f"the agent the replica will use, so adapting the one you were "
            f"handed means returning it too: "
            f"`def agent(built, index): built.boundaries = Mine(); return built`."
        )
    try:
        check_agent(adapted)
    except TypeError as exc:
        raise TypeError(
            f"the agent= factory returned a {type(adapted).__name__} for "
            f"replica {index}, which is not a ttrl.Agent: {exc} Return the "
            f"agent you were handed, adapted, or an object carrying the whole "
            f"Agent surface."
        ) from None
    return adapted


def _source(config: TrainConfig, source, index: int):
    """This replica's step source: the caller's factory, or the env builder.

    `builders.build_env` checks the env contract on the stack it builds
    (`ttrl.check_env_contract`), and that check stays on the builder path
    alone: it asks what an OBSERVATION has to be, which a source replaying a
    logged dataset or driving a robot may answer differently while still
    stepping a run correctly. `ttrl.train.loop.StepSource` is what a source is
    held to instead, and is checked here, by name, before the first frame.
    """
    if source is None:
        return builders.build_env(config.env)
    built = source(config.env, index)
    missing = [
        name
        for name in ("obs_spec", "action_spec", "num_envs", "reset", "step",
                     "close", "save_policy_state", "load_policy_state",
                     "save_run_state", "load_run_state")
        if not hasattr(built, name)
    ]
    if missing:
        raise TypeError(
            f"the source= factory returned a "
            f"{type(built).__name__} for replica {index}, which is not a "
            f"ttrl.StepSource: it is missing {missing}. The first five are "
            f"what ttrl.train.loop.run_loop calls; close() is the teardown "
            f"ladder's, and the two state pairs are what a checkpoint carries "
            f"and a resume restores. A source that carries neither kind of "
            f"state returns [] from both saves and ignores both loads, which "
            f"is what says so rather than leaving a run to guess."
        )
    return built


def _devices(config: TrainConfig, device, profile, stack):
    if config.device.chips == 1:
        if device is None or profile is None:
            raise ValueError(
                "config.device.chips is 1, so the caller owns the device and must "
                "pass both device and profile. Open one with ttrl.engine.acquire, "
                "or set config.device.chips to let the Trainer open its own."
            )
        return [device], [profile]

    if device is not None or profile is not None:
        raise ValueError(
            f"config.device.chips is {config.device.chips!r}, so the Trainer opens "
            "one independent device per chip itself; pass device=None and "
            "profile=None. A handle opened here would hold a chip the replicas "
            "are about to open."
        )

    # Deferred: ttrl.engine.mesh imports ttnn, and this module loads before
    # chip scoping.
    from ttrl.engine.mesh import open_independent_devices_for_chips

    engine_cfg = config.device.engine_config()
    chips = resolve_run_chips(config.device.chips)
    devices = stack.enter_context(
        open_independent_devices_for_chips(chips, cfg=engine_cfg)
    )
    return list(devices), [probe_profile(dev, engine_cfg) for dev in devices]


def resolve_run_chips(chips):
    """The physical chip ids a run with this `chips` setting will open.

    Resolved against the VISIBLE set rather than every driver node. A process
    scoped to a subset cannot open the others, so `chips: "all"` resolved from
    the driver would name them and then be refused by
    require_exact_visible_scope, failing a run that was given everything it
    asked for.
    """
    # Deferred: ttrl.engine.mesh imports ttnn, and this module loads before
    # chip scoping.
    from ttrl.engine.mesh import resolve_chips

    visible = visible_chip_ids()
    return resolve_chips(chips, physical_chips() if visible is None else visible)


def _scaled_algo_config(config: TrainConfig, n: int):
    """The algorithm's config as this run's replicas should each see it."""
    if config.algo_cfg is None:
        return None
    return config.algo_cfg.scale_for_replicas(n)


def _param_sync(config: TrainConfig, replicas):
    """Exact fp32 parameter averaging across the replicas, or None at n=1.

    None rather than a one-replica `ParamSync`: averaging one replica with
    itself is a no-op that still costs a full host round trip of every
    parameter on every sync.
    """
    if len(replicas) == 1:
        return None
    # Deferred: ttrl.engine.mesh imports ttnn, and this module loads before
    # chip scoping.
    from ttrl.engine.mesh import ParamSync

    return ParamSync(
        modules=[replica.agent for replica in replicas],
        sync_every=config.device.sync_every,
    )
