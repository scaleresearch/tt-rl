"""Register/resolve an algorithm by name, so adding one edits no core file.

`ttrl.nets.registry.Registry` already carries the duplicate-name refusal and
the unknown-name message listing what is registered. `module=` is what points
that message at this file rather than at the nets one, so nothing here is
reimplemented.

THE ENTRIES ARE TARGET STRINGS, as `ttrl.envs.registry`'s are, and for a
sharper version of the same reason. A registry that imports the modules
holding its builders answers `name in ALGOS`, `ALGOS.names()` and `len(ALGOS)`
by importing all three algorithms, and those import ttnn. Every one of those
questions is asked before a chip is scoped: `ttrl.train.run` asks whether the
configured name is registered, and a third-party package guards its own
registration with `if "mine" not in ttrl.ALGOS` because that is what this
library's own code looks like. An unscoped ttnn import maps hugepages and
takes the driver mutex for every chip on the board, which on a shared box
means seeing chips another process holds, with no error anywhere.

A target string is resolved on first use, so asking WHICH algorithms exist
costs no import at all, and asking for one imports that one rather than all
three.
"""

from __future__ import annotations

from ttrl.nets.registry import Registry
from ttrl.spaces import ActionSpec, ObsSpec

__all__ = ["ALGOS", "resolve_and_build"]

#: `builder(obs_spec, action_spec, device, **kwargs) -> Agent`
ALGOS = Registry("algos", module="ttrl.algos.registry", allow_targets=True)

ALGOS.register("dqn", "ttrl.algos.dqn.agent:DQN")
ALGOS.register("ppo", "ttrl.algos.ppo.agent:PPO")
ALGOS.register("sac", "ttrl.algos.sac.agent:SAC")


def resolve_and_build(name: str, obs_spec: ObsSpec, action_spec: ActionSpec,
                      device, **kwargs):
    """Resolve an algorithm by name and construct it. Unknown names raise.

    Not called `build_agent`: `ttrl.train.builders.build_agent` is the
    config-driven one that also runs `check_agent`, and two functions of that
    name told a reader of a traceback nothing about which one they were in.
    This is the registry's half, and it checks nothing.

    Imports the algorithm it names, and therefore ttnn: scope the process
    first. Asking whether a name is registered does not; see `ALGOS` above.
    """
    return ALGOS.resolve(name)(obs_spec, action_spec, device, **kwargs)
