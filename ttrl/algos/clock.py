"""The replay-ratio clock: how much gradient work the frames so far are worth.

Its own module, and deliberately free of ttnn. The arithmetic here is plain
ints and floats, and it is the one piece of the off-policy cadence a host test
can check exactly. Living in `ttrl.algos.offpolicy` put it behind that
module's `ttrl.engine` imports, so every probe of the replay ratio had to
reach ttnn first and therefore had to be scoped to a chip before it could run
at all.

ASKING AND ACCRUING ARE SEPARATE CALLS. `advance` moves the clock, `due` and
`owed` only read it. They used to be one call, a `due(frame)` that accrued as
a side effect of being asked, and that is what made a multi-replica run wrong:
a caller that asked ONE replica on behalf of several left every other
replica's clock at zero credit, so each of them took `max(1, owed) == 1`
gradient step per round however much data the round carried. On two chips
that arithmetic leaves replica 1 with 12.5% of replica 0's gradient work. A
predicate that accrues cannot be asked on someone else's behalf, and nothing
in its signature says so, so the two operations are now named separately.
"""

from __future__ import annotations

__all__ = ["UpdateClock"]


class UpdateClock:
    """How many gradient steps the frames collected so far are worth.

    `train_freq` is frames per gradient step, so a vector env that advances
    `n_envs` frames per step is worth `n_envs / train_freq` of one. Testing
    `frame % train_freq == 0` instead costs a factor of `n_envs` in the
    realised replay ratio whenever `n_envs` does not divide `train_freq`: at
    `n_envs=3, train_freq=4` only every twelfth frame is a multiple, so the run
    takes one gradient step per 12 frames where its config asked for one per 4.
    Nothing reports it; the run simply learns three times less per frame.

    Credit accrues from the frame number rather than from the number of calls,
    so advancing twice to the same frame accrues once, and it accrues only past
    the warm-up, so a run does not owe `warmup / train_freq` steps the moment
    it ends.
    """

    def __init__(self, train_freq: int, warmup: int):
        if train_freq < 1:
            raise ValueError(f"UpdateClock: train_freq must be >= 1, got {train_freq}")
        if warmup < 0:
            raise ValueError(f"UpdateClock: warmup must be >= 0, got {warmup}")
        self.train_freq = int(train_freq)
        self.warmup = int(warmup)
        self._frame = 0
        self._credit = 0.0

    def advance(self, frame: int) -> None:
        """Accrue the gradient-step credit the frames up to `frame` are worth.

        Idempotent in the frame number: the credit comes from the distance
        moved, so advancing to a frame already reached accrues nothing and
        every replica advanced to the same frame holds the same credit.
        """
        start = max(self._frame, self.warmup)
        if frame > start:
            self._credit += (frame - start) / self.train_freq
        self._frame = max(self._frame, int(frame))

    def skip(self, frame: int) -> None:
        """Advance past frames the agent could not have learned from.

        A frame collected while the buffer holds no complete window to sample
        is not a frame whose gradient step was missed, so it owes none. Without
        this the credit for every such frame is kept and taken in one burst the
        moment the buffer fills, which is unbounded work in a single call.
        """
        self._frame = max(self._frame, int(frame))

    @property
    def due(self) -> bool:
        """Whether at least one gradient step is owed. Reads, never accrues."""
        return self._credit >= 1.0

    @property
    def owed(self) -> int:
        """Whole gradient steps owed. At least one once `due` is true."""
        return int(self._credit)

    def consume(self, steps: int) -> None:
        """Record `steps` gradient steps as taken.

        The remainder is kept rather than dropped: a run whose frames per step
        do not divide `train_freq` would otherwise lose the fraction on every
        round and drift below its configured ratio.
        """
        self._credit -= float(steps)

    def state_dict(self) -> dict:
        return {"frame": self._frame, "credit": self._credit}

    def load_state_dict(self, state: dict) -> None:
        self._frame = int(state["frame"])
        self._credit = float(state["credit"])
