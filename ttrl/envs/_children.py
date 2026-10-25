"""The child-process half of `AsyncVectorEnv`: spawn, talk, shut down.

Everything here is about processes and pipes. Nothing here knows what a
command means, what an observation looks like or which lane owes a reset;
that is `AsyncVectorEnv`'s half. The split exists because the two halves fail
differently. A mistake in the autoreset bookkeeping produces wrong numbers, a
mistake in this file leaves a subprocess alive holding a simulator instance
while a long run keeps going, and the escalation that prevents that is worth
reading on its own.

The one invariant that crosses the seam: every command sent must have its
reply read before the next command goes out, or the lane desynchronises. That
is why `collect` reads every lane even when an early lane has already failed.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Sequence

from ttrl.envs._worker import LaneError, worker_main

__all__ = ["LaneGroup", "CLOSE_TIMEOUT_S"]

#: Seconds to wait for a child to exit after it is asked to close, before it
#: is terminated. A child holding an ALE instance that outlives its parent
#: wedges the machine, so shutdown escalates rather than waiting forever.
CLOSE_TIMEOUT_S = 10.0


class LaneGroup:
    """One daemon subprocess per lane, and the pipe to each.

    `env_fns` are pickled to the children under the default "spawn" start
    method, so each must be a module-level function or another picklable
    callable; a lambda or local closure is not. `ttrl.envs.registry` exists
    partly for this: a small picklable builder keyed by name. Pass
    `context="fork"` to use closures, accepting that a forked child inherits
    the parent's threads and any device handle it holds.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        context: str = "spawn",
    ):
        ctx = mp.get_context(context)
        self._conns = []
        self._procs = []
        self._closed = False
        for i, fn in enumerate(env_fns):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=worker_main, args=(child_conn, i, fn), daemon=True
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

    @property
    def num_envs(self) -> int:
        return len(self._procs)

    @property
    def closed(self) -> bool:
        return self._closed

    def require_open(self) -> None:
        if self._closed:
            raise RuntimeError("AsyncVectorEnv is closed")

    def send(self, lane: int, command: str, payload: Any) -> None:
        self._conns[lane].send((command, payload))

    def broadcast(self, command: str, payload: Any = None) -> None:
        for lane in range(self.num_envs):
            self.send(lane, command, payload)

    def collect(self, what: str) -> list[Any]:
        """Read one reply per lane, reporting every lane's failure together.

        Every lane is read before anything is raised, so one lane's failure
        never leaves a sibling's reply unread in the pipe.
        """
        results: list[Any] = [None] * self.num_envs
        failures: list[BaseException] = []
        for lane, conn in enumerate(self._conns):
            try:
                status, payload = conn.recv()
            except (EOFError, OSError) as exc:
                failures.append(
                    LaneError(
                        lane,
                        f"the subprocess died without replying during {what} "
                        f"(exitcode {self._procs[lane].exitcode}): {exc!r}",
                    )
                )
                continue
            if status == "error":
                failures.append(payload)
            else:
                results[lane] = payload
        if failures:
            lanes = [getattr(e, "lane", -1) for e in failures]
            raise ExceptionGroup(
                f"AsyncVectorEnv {what} failed in lanes {lanes} of {self.num_envs}",
                failures,
            )
        return results

    def close(self) -> None:
        """Ask every child to exit, then make sure every child has exited.

        A child that has already died cannot be asked anything, and that is
        not an error during shutdown; what would be an error is leaving a
        subprocess alive holding a simulator instance.
        """
        if self._closed:
            return
        self._closed = True
        for conn, proc in zip(self._conns, self._procs):
            if proc.is_alive():
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass  # the child is already gone; nothing to ask
        for proc in self._procs:
            proc.join(CLOSE_TIMEOUT_S)
            if proc.is_alive():
                proc.terminate()
                proc.join(CLOSE_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
                proc.join()
        # Pipes close only once no child can still be writing to one.
        for conn in self._conns:
            conn.close()
        survivors = [i for i, p in enumerate(self._procs) if p.is_alive()]
        if survivors:
            raise RuntimeError(
                f"AsyncVectorEnv.close: lanes {survivors} would not exit after "
                f"close, terminate and kill. Their simulator instances are leaked."
            )
