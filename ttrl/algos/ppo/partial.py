"""The half-collected rollout a PPO checkpoint carries, as plain arrays.

WHY THIS EXISTS. PPO's checkpoint used to hold weights, moments, counts, RNG
and autoreset phase but no rollout, so a file written mid-rollout dropped
every step sitting in the buffer. Measured: 1024 frames at a 64-step rollout
take 16 updates uninterrupted and 16 across a split at 512, which is rollout
aligned; split at 544, with 8 steps in the buffer, they took 15. The resumed
leg entered with an empty buffer and the run reported itself as a
continuation. It went unnoticed because `checkpoint_interval` is usually a
multiple of the rollout size, and because the usual way to land off a
boundary is the final save of a run whose `total_frames` is not a multiple of
one.

WHY IT IS NOT OPT-IN, unlike `ttrl.buffers.persist`. That buffer is opt-in
because a 1M-transition Atari replay of stacked frames is 28 GB. A PPO
rollout is `n_steps x n_envs` of experience and the arrays below are TRIMMED
to the write cursor, so a checkpoint written on a boundary carries zero-length
arrays. Measured on a Pong-shaped config (128 steps x 8 lanes of 4x84x84
uint8, which is what every Atari PPO config in `configs/` collects): 3,354
bytes with the buffer empty, 1.8 MB with 8 steps in it, and 28.7 MB at the
worst case of 127. The whole member is three orders of magnitude under the
replay buffer whose size made that one opt-in, so there is nothing here for a
run to decide.

THE ACTING CARRY IS PART OF THE ROLLOUT, and the seam is what bounds it. The
stored `hidden` rows are the state entering each sequence's first step, so a
mid-sequence resume loads a sequence whose first steps were recorded with a
real memory. What it must NOT do is carry that memory into the first step
after the resume, which belongs to a different episode: the environment is
reset before that step is taken. `_close_the_seam` marks the boundary, and the
act that consumes the restored carry masks it away on exactly that step. The
carry is still loaded and still checked against this policy's memories,
because a file whose memories are not this run's is the wrong file.

THE CHECK ON THE WAY BACK IN IS GEOMETRY. A rollout is time-major and loads
into `[:steps]`, so there is no ring arithmetic to get wrong, but a buffer of
a different `n_steps`, lane count, sequence length or observation leaves is a
different store and is refused by name.

`_last_next_obs` is deliberately absent. It is the bootstrap observation for
`compute_gae`, and it is rewritten by the first `observe` after a resume,
which always precedes the next update: a checkpoint is written after the
round's update, so the buffer it carries is never full.
"""

from __future__ import annotations

import numpy as np

from ttrl.engine.tensor import restore_in_place

__all__ = ["load_rollout_state", "rollout_state"]

#: The per-step arrays, each `[n_steps, n_envs, ...]` and each trimmed to the
#: write cursor. Enumerated here once; `load_rollout_state` walks the same
#: tuple, so a field added to the buffer and to this list round-trips on both
#: sides or on neither.
STEP_FIELDS = ("actions", "old_logp", "values", "rewards", "dones", "valid",
               "starts")

#: What the stored arrays' meaning depends on. `seq_len` is None outside the
#: recurrent case.
GEOMETRY = ("n_steps", "n_envs", "seq_len")


def rollout_state(agent) -> dict:
    """Every step in `agent`'s rollout buffer, its cursor and its carry.

    Copies, and trimmed to the cursor: the slice is a view until it is
    written, and the agent keeps training after the snapshot is taken.
    """
    buffer = agent.buffer
    steps = int(buffer._t)
    state = {
        "geometry": {name: _geometry_value(buffer, name) for name in GEOMETRY},
        "leaves": list(buffer.obs_spec.names),
        "steps": steps,
        "obs": {name: array[:steps].copy()
                for name, array in buffer.obs.items()},
        "carried_boundary": buffer._carried_boundary.copy(),
        "batch_rng": buffer._rng.bit_generator.state,
        "hidden": _hidden(buffer, steps),
        "acting_hidden": _acting_hidden(agent),
    }
    for name in STEP_FIELDS:
        state[name] = getattr(buffer, name)[:steps].copy()
    return state


def load_rollout_state(agent, state: dict) -> None:
    """Restore the rollout `rollout_state` wrote, or refuse by name.

    In place into the buffer's own arrays, after a `reset`, so every slot the
    file does not carry is the one a fresh rollout would have.
    """
    buffer = agent.buffer
    _check_geometry(buffer, state)
    steps = int(state["steps"])
    if not 0 <= steps < buffer.n_steps:
        raise ValueError(
            f"this checkpoint's PPO rollout holds {steps} steps and the buffer "
            f"takes {buffer.n_steps}, exclusive: a checkpoint is written after "
            f"the round's update, so a full rollout cannot have been stored "
            f"and this file was not written by this code. Train afresh."
        )
    buffer.reset()
    for name in buffer.obs_spec.names:
        _copy(buffer.obs[name][:steps], state["obs"][name], f"obs[{name!r}]")
    for name in STEP_FIELDS:
        _copy(getattr(buffer, name)[:steps], state[name], name)
    _copy(buffer._carried_boundary, state["carried_boundary"],
          "carried_boundary")
    buffer._rng.bit_generator.state = state["batch_rng"]
    for name, array in state["hidden"].items():
        _copy(buffer.hidden[name][:len(array)], array, f"hidden[{name!r}]")
    buffer._t = steps
    _close_the_seam(buffer, steps)
    _load_acting_hidden(agent, state["acting_hidden"])


def _close_the_seam(buffer, steps: int) -> None:
    """Cut the stored rollout off from the steps collected after the resume.

    THE ENVIRONMENT THIS LOADS INTO IS RESET. `ttrl.train.loop.run_loop`
    resets every replica before its first round and this file carries no
    simulator state, so the step after the last stored one is the first step
    of a NEW episode in every lane. Nothing above marks that, and three things
    then read across it: `compute_gae` bootstraps the last stored step from
    `values[steps]` and chains the new episode's advantages back through the
    whole rollout; `starts[steps]` reads 0, so the sequence replay carries the
    old episode's memory into the new one; and the acting carry below is
    applied to the new episode's first observation.

    The last stored step becomes a PADDING SLOT, which is the buffer's own
    word for a step whose successor is not in the rollout: it takes no
    gradient and `compute_gae` zeroes the chain at it, so the step before it
    bootstraps from `values[steps - 1]`, the honest value of the state it
    really led to. One slot per lane per resume loses its gradient and nothing
    is fabricated. Closing it as a termination instead would have invented a
    terminal state, and leaving it open was the splice.

    The carried boundary goes to 1 for every lane, so the first step after the
    resume is recorded as beginning an episode and `_load_acting_hidden`,
    which reads this same flag, masks the restored memory away on the act that
    consumes it.
    """
    if steps:
        buffer.valid[steps - 1] = 0.0
    buffer._carried_boundary[:] = 1.0


# -- the recurrent half ----------------------------------------------------


def _hidden(buffer, steps: int) -> dict:
    """The sequence-start states recorded so far, one row per chunk reached.

    Rounded UP, and so not `ttrl.buffers.sequences.chunk_count`, which refuses
    a remainder because a short sequence is a second graph shape. A partial
    rollout is exactly that remainder: the sequence it stopped inside has its
    first step's state recorded and is finished after the resume.
    """
    if not buffer.recurrent:
        return {}
    reached = -(-steps // buffer.seq_len)
    return {name: array[:reached].copy()
            for name, array in buffer.hidden.items()}


def _acting_hidden(agent) -> dict:
    """The live training carry, per memory, read to the host.

    Only the training width and the stochastic purpose: an evaluation carry
    belongs to episodes of a different env instance and is rebuilt on its
    first use.
    """
    runtime = agent._recurrent
    if runtime is None:
        return {}
    carries = runtime.carries(agent.n_envs, deterministic=False)
    return {name: carry.numpy().copy() for name, carry in carries.items()}


def _load_acting_hidden(agent, stored: dict) -> None:
    runtime = agent._recurrent
    if runtime is None:
        if stored:
            raise ValueError(
                f"this checkpoint's PPO rollout carries acting hidden state "
                f"for {sorted(stored)} and this run's policy has no memory. "
                f"Resume with the algorithm config the checkpoint stores "
                f"(ttrl.config_from_checkpoint prints it), or start afresh."
            )
        return
    runtime.zero()
    carries = runtime.carries(agent.n_envs, deterministic=False)
    missing = sorted(set(carries) - set(stored))
    if missing:
        raise ValueError(
            f"this checkpoint's PPO rollout carries no acting hidden state "
            f"for {missing}, which this policy's memories need. Train afresh."
        )
    for name, carry in carries.items():
        array = np.asarray(stored[name], dtype=np.float32)
        if array.shape != (carry.lanes, carry.size):
            raise ValueError(
                f"this checkpoint's PPO acting hidden state for {name!r} is "
                f"shaped {array.shape} and this run's carry is "
                f"{(carry.lanes, carry.size)}. Resume with the algorithm "
                f"config the checkpoint stores "
                f"(ttrl.config_from_checkpoint prints it), or start afresh."
            )
        restore_in_place(carry.prev.data, array)
    # The flag the NEXT act masks with is the last stored step's boundary,
    # which is the one the buffer carries across rollouts.
    runtime._boundary = np.asarray(
        agent.buffer._carried_boundary, dtype=np.float32).copy()


# -- the checks ------------------------------------------------------------


def _geometry_value(buffer, name: str):
    value = getattr(buffer, name)
    return None if value is None else int(value)


def _check_geometry(buffer, state: dict) -> None:
    stored = {k: (None if v is None else int(v))
              for k, v in state["geometry"].items()}
    live = {name: _geometry_value(buffer, name) for name in GEOMETRY}
    if stored != live:
        differ = sorted(k for k in live if stored.get(k) != live[k])
        raise ValueError(
            f"this checkpoint's PPO rollout differs from this run's in "
            f"{differ}: the file has {stored} and this run has {live}. A "
            f"rollout is laid out by step and lane and its sequences are cut "
            f"at seq_len, so these steps cannot be loaded into a buffer of "
            f"another shape. Resume with the algorithm config the checkpoint "
            f"stores (ttrl.config_from_checkpoint prints it), or start afresh."
        )
    leaves = list(state["leaves"])
    if leaves != list(buffer.obs_spec.names):
        raise ValueError(
            f"this checkpoint's PPO rollout stores observation leaves "
            f"{leaves} and this run's observation has "
            f"{list(buffer.obs_spec.names)}. Resume under the env the "
            f"checkpoint was written with, or start afresh."
        )


def _copy(destination: np.ndarray, source, name: str) -> None:
    array = np.asarray(source)
    if array.shape != destination.shape:
        raise ValueError(
            f"this checkpoint's PPO rollout {name} is shaped {array.shape} "
            f"and this run's slot for it is {destination.shape}. Train afresh."
        )
    destination[...] = array
