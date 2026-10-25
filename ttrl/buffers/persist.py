"""A replay buffer as plain arrays, and the geometry it must come back into.

WHY THIS EXISTS. Every other part of a run's state is checkpointed: weights,
optimizer moments, the exploration RNG, the frame and update counts, the
environment's running statistics. The transitions were not, so a resumed
off-policy run restarted with an EMPTY buffer while its epsilon schedule and
its update clock carried on from where they stopped. The first gradient step
after such a resume trains on a handful of transitions at a fully annealed
exploration rate, which does not fail and does not warn: it produces a worse
policy than the uninterrupted run and reports itself as a continuation.

WHY IT IS OPT-IN. A 1M-transition Atari buffer of stacked uint8 frames is
28 GB, and a run checkpointing every 100k frames would fill a disk in an
afternoon. So the run states whether it wants this
(`TrainConfig.checkpoint_replay`), and `ttrl.train.resume` refuses a warm
resume from a file that does not carry one rather than resuming cold.

A SEPARATE MODULE, not methods on `ReplayBuffer`, because the seam is real:
the buffer owns the ring arithmetic and the sampling semantics, and this owns
one wire format for them plus the compatibility check on the way back in. It
reads the buffer's private cursors deliberately, being the one place that
serialises them.

THE CHECK ON THE WAY BACK IN IS GEOMETRY, NOT A VERSION. Loading a lane's
columns into a ring of a different width, or into a different n-step, would
put real data at indices that mean something else, and the ring carries no
mark that would let a later read notice. So every field the indexing depends
on is compared and a mismatch is refused by name.
"""

from __future__ import annotations

import numpy as np

__all__ = ["load_replay_state", "owns_replay", "replay_state"]

#: The scalars the ring arithmetic and the n-step read are derived from. A
#: stored buffer whose geometry differs in any of them cannot be loaded here.
GEOMETRY = ("per_lane_capacity", "n_envs", "n_step")


def replay_state(buffer) -> dict:
    """Every transition in `buffer`, its cursors, and its sampling state.

    The arrays go out whole, at capacity, rather than trimmed to `size`: a
    trimmed ring would have to be re-laid-out on the way back in, and the
    column a transition sits in is what its successor read depends on.

    COPIES, not views of the live ring. A view costs nothing until the caller
    keeps training with the snapshot in hand, and then the ring's arrays move
    on while the boundary table, which has to be rebuilt as arrays either
    way, stays as it was. The result is a snapshot whose `ends` marks
    boundaries its table has no entry for, which surfaces thousands of steps
    later as a successor read for a column that "ends an episode but carries
    no final observation". Measured by exactly that failure. The copy is also
    not where the memory goes: `save_checkpoint` serialises the npz into a
    host buffer, so the opt-in case already holds the transitions twice.
    """
    state = {
        "geometry": {name: int(getattr(buffer, name)) for name in GEOMETRY},
        "gamma": float(buffer.gamma),
        "leaves": list(buffer.obs_spec.names),
        "obs": {name: array.copy() for name, array in buffer.obs.items()},
        "actions": buffer.actions.copy(),
        "rewards": buffer.rewards.copy(),
        "dones": buffer.dones.copy(),
        "ends": buffer.ends.copy(),
        "pos": buffer._pos.copy(),
        "full": buffer._full.copy(),
        "boundaries": buffer._final_obs.state(),
        "open_successors": {str(lane): dict(obs)
                            for lane, obs in buffer._open.items()},
        "rng_state": buffer._rng.bit_generator.state,
    }
    _refuse_an_unrecorded_seam(buffer)
    priorities = buffer.priority_state()
    if priorities is not None:
        state["priorities"] = priorities
    return state


def _refuse_an_unrecorded_seam(buffer) -> None:
    """A lane mid-episode whose successor nothing recorded.

    `_close_the_seam` can only close a lane it knows what followed, and
    `note_open_successor` is what records that, per write. Every write
    `ttrl.algos.offpolicy.ReplayWriter` makes calls it; `add_batch` does not,
    so a writer that fills a buffer that way would snapshot a seam no reader
    can close. Caught where the snapshot is taken, because the alternative is
    a resume that splices and says nothing.
    """
    newest = np.where(buffer._pos > 0, buffer._pos - 1,
                      buffer.per_lane_capacity - 1)
    written = (buffer._pos > 0) | buffer._full
    open_lanes = [
        lane for lane in range(buffer.n_envs)
        if written[lane] and not buffer.ends[lane, newest[lane]]
        and lane not in buffer._open
    ]
    if open_lanes:
        raise ValueError(
            f"lanes {open_lanes} of this replay buffer are mid-episode and "
            f"nothing recorded what followed their newest transition, so a "
            f"resume could neither pair it with its successor nor end the "
            f"episode there. Every write through "
            f"ttrl.algos.offpolicy.ReplayWriter records it; add_batch does "
            f"not. Write through the ReplayWriter, or call "
            f"note_open_successor beside each add."
        )


def load_replay_state(buffer, state: dict) -> None:
    """Restore `buffer` from `replay_state`, or refuse by name.

    In place into the buffer's existing arrays, so nothing holding a view of
    one keeps reading the old ring.
    """
    _check_geometry(buffer, state)
    for name in buffer.obs_spec.names:
        _copy(buffer.obs[name], state["obs"][name], f"obs[{name!r}]")
    for name in ("actions", "rewards", "dones", "ends"):
        _copy(getattr(buffer, name), state[name], name)
    _copy(buffer._pos, state["pos"], "pos")
    _copy(buffer._full, state["full"], "full")
    buffer._final_obs.load(state["boundaries"])
    buffer._rng.bit_generator.state = state["rng_state"]
    stored = state.get("priorities")
    if (stored is None) != (buffer.priority_state() is None):
        raise ValueError(
            f"this checkpoint's replay buffer "
            f"{'carries' if stored is not None else 'carries no'} priorities "
            f"and this run's buffer is "
            f"{'prioritized' if stored is not None else 'uniform'}"
            f". Prioritized and uniform replay converge to different policies, "
            f"so the two are not interchangeable mid-run: resume with the "
            f"algorithm config the checkpoint stores "
            f"(ttrl.config_from_checkpoint prints it), or start afresh."
        )
    if stored is not None:
        buffer.load_priority_state(stored)
    _close_the_seam(buffer, state)


def _close_the_seam(buffer, state: dict) -> None:
    """End the episodes the stored transitions were in the middle of.

    THE ENVIRONMENT THIS LOADS INTO IS RESET. `ttrl.train.loop.run_loop`
    resets every replica before its first round, a resume included, and no
    part of a Gymnasium simulator's state is in this file, so the episode a
    lane was halfway through does not continue. What continues is the ring,
    and the newest column of every unfinished lane derives its successor from
    the next column, which the first step after the reset fills with a new
    episode's first observation. Left alone, each such lane contributes one
    transition the environment never made, plus every n-step window that
    reaches across the seam.

    So the seam is closed rather than crossed: `close_open_episodes` marks
    those columns truncated against the successors the writer recorded. The
    successors are what makes it exact, which is why a file without them is
    refused instead of being closed on a guess.
    """
    if "open_successors" not in state:
        raise ValueError(
            "this checkpoint's replay buffer carries no 'open_successors', so "
            "nothing in it says what followed the last transition of each "
            "lane that was mid-episode. A resume resets every environment, so "
            "those transitions would be paired with the first observation of "
            "a new episode and trained on as if the environment had made that "
            "step. The checkpoint format is pre-release and carries no "
            "migrations, so nothing can add the member to this file after the "
            "fact. Resume from a file written by this build, score these "
            "weights with ttrl.evaluate_checkpoint, which reads no "
            "experience, or start the run afresh."
        )
    buffer._open = {int(lane): {name: np.asarray(array) for name, array in obs.items()}
                    for lane, obs in state["open_successors"].items()}
    buffer.close_open_episodes()


def _check_geometry(buffer, state: dict) -> None:
    stored = {k: int(v) for k, v in state["geometry"].items()}
    live = {name: int(getattr(buffer, name)) for name in GEOMETRY}
    if stored != live:
        differ = sorted(k for k in live if stored.get(k) != live[k])
        raise ValueError(
            f"this checkpoint's replay buffer differs from this run's in "
            f"{differ}: the file has {stored} and this run has {live}. A "
            f"lane's columns are consecutive steps of one stream and a "
            f"transition's successor is the next column, so loading these "
            f"transitions into a ring of another shape would pair each one "
            f"with a successor it never had, with nothing able to detect it "
            f"afterwards. Resume with the config the checkpoint stores "
            f"(ttrl.config_from_checkpoint prints it), or start afresh."
        )
    leaves = list(state["leaves"])
    if leaves != list(buffer.obs_spec.names):
        raise ValueError(
            f"this checkpoint's replay buffer stores observation leaves "
            f"{leaves} and this run's observation has "
            f"{list(buffer.obs_spec.names)}. The stored transitions describe "
            f"another observation, so there is no slot in this buffer to load "
            f"them into. Resume with the env config the checkpoint stores "
            f"(ttrl.config_from_checkpoint prints it), or start afresh."
        )
    if float(state["gamma"]) != float(buffer.gamma):
        raise ValueError(
            f"this checkpoint's replay buffer discounts at gamma="
            f"{state['gamma']} and this run's at gamma={buffer.gamma}. The "
            f"n-step return of every stored transition was accumulated at the "
            f"stored one, and nothing can re-accumulate it from what the file "
            f"holds. Resume with the algorithm config the checkpoint stores "
            f"(ttrl.config_from_checkpoint prints it), or start afresh."
        )


def _copy(destination: np.ndarray, source, name: str) -> None:
    array = np.asarray(source)
    if array.shape != destination.shape:
        raise ValueError(
            f"this checkpoint's replay {name} is shaped {array.shape} and "
            f"this run's is {destination.shape}. Resume with the config the "
            f"checkpoint stores (ttrl.config_from_checkpoint prints it), or "
            f"start afresh."
        )
    destination[...] = array


def owns_replay(agent) -> bool:
    """Whether `agent` has experience a checkpoint could carry.

    A duck check rather than a registry of algorithm names: an off-policy
    agent exposes this pair of methods, an on-policy one has no stored
    experience to continue from (its rollout is discarded at every iteration
    by construction), and a new algorithm says which it is by whether it
    implements them. The `ttrl.train` layer asks this rather than asking what
    `config.algo` is.
    """
    return all(callable(getattr(agent, name, None))
               for name in ("replay_state", "load_replay_state"))
