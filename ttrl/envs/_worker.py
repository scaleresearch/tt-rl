"""The child-process side of `ttrl.envs.vector_async.AsyncVectorEnv`.

A module-level function rather than a closure, because the "spawn" start
method pickles the target. The autoreset decision deliberately stays in the
parent, so a lane here only ever does what it is told and the two vector
implementations produce identical trajectories from the same seed.
"""

from __future__ import annotations

import traceback

from ttrl.envs.wrappers.core import (
    restore_env,
    restore_policy_state,
    snapshot_env,
    snapshot_policy_state,
)

__all__ = ["LaneError", "worker_main"]


class LaneError(RuntimeError):
    """A failure raised inside a lane's subprocess, carried to the parent.

    The child's traceback travels as text: formatting it in the child is the
    only way the parent can report where the failure happened, and a string
    is always picklable where an arbitrary exception object is not.
    """

    def __init__(self, lane: int, child_traceback: str):
        super().__init__(f"lane {lane} raised in its subprocess:\n{child_traceback}")
        self.lane = lane
        self.child_traceback = child_traceback

    # Exceptions are rebuilt from `args` by default, which for this one is the
    # formatted message rather than the two fields. Without this the error
    # fails to unpickle in the parent and masks the failure it describes.
    def __reduce__(self):
        return (LaneError, (self.lane, self.child_traceback))


def worker_main(conn, lane: int, env_fn) -> None:
    try:
        env = env_fn()
    except BaseException:
        conn.send(("error", LaneError(lane, traceback.format_exc())))
        conn.close()
        return
    conn.send(("spec", (env.obs_spec, env.action_spec)))
    try:
        while True:
            command, payload = conn.recv()
            if command == "close":
                # No acknowledgement: the parent is already joining, and a
                # reply racing a closed pipe raises BrokenPipeError here.
                return
            # The except below transports the failure to the parent, which
            # re-raises it. Nothing is swallowed and no alternative path runs.
            try:
                if command == "step":
                    result = env.step(payload)
                elif command == "reset":
                    result = env.reset(seed=payload)
                elif command == "get_state":
                    result = snapshot_env(env)
                elif command == "set_state":
                    restore_env(env, payload)
                    result = None
                elif command == "get_policy_state":
                    # Selected in the child, where the live wrapper objects and
                    # their POLICY_STATE declarations are.
                    result = snapshot_policy_state(env)
                elif command == "set_policy_state":
                    restore_policy_state(env, payload)
                    result = None
                else:
                    raise ValueError(f"unknown worker command {command!r}")
            except BaseException:
                conn.send(("error", LaneError(lane, traceback.format_exc())))
                continue
            conn.send(("ok", result))
    finally:
        env.close()
        conn.close()
