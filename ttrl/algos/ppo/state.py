"""PPO's checkpoint halves.

Split out of the agent because what a resume must restore is a list that grows
every time the algorithm gains a piece of run state, and the agent file is
where that list is least visible. The two functions here are the whole
contract: `WEIGHT_KEYS` and `RUN_KEYS` name the halves `ttrl.algos.checkpoint`
validates against, so a key added to one and not the other fails loudly at
load time.

THE HALF-COLLECTED ROLLOUT IS IN THE RUN HALF, under `rollout`, and
`ttrl.algos.ppo.partial` is its wire format and the measurement that put it
there. It used to be discarded, which cost a resumed run one gradient step
per non-boundary checkpoint, silently. The hidden state it was collected with
goes with it for the same reason, so a mid-sequence resume continues the
sequence rather than splicing a blank memory into it.
"""

from __future__ import annotations

from ttrl.algos.checkpoint import assemble, validate
from ttrl.algos.ppo.partial import load_rollout_state, rollout_state

__all__ = ["RUN_KEYS", "WEIGHT_KEYS", "load_into", "state_of"]

WEIGHT_KEYS = ("net",)
RUN_KEYS = ("optimizer", "frame_count", "update_count", "rng_state",
            "autoreset", "rollout")


def state_of(agent, *, weights_only: bool) -> dict:
    return assemble(
        agent.cfg,
        {"net": agent.net.state_dict()},
        {
            "optimizer": agent.optimizer.state_dict(),
            "frame_count": agent.frame_count,
            "update_count": agent.update_count,
            "rng_state": agent._rng.bit_generator.state,
            "autoreset": agent._autoreset.state_dict(),
            "rollout": rollout_state(agent),
        },
        weights_only=weights_only,
    )


def load_into(agent, state: dict, *, weights_only: bool) -> None:
    load_run = validate(state, algo="PPO", config=agent.cfg,
                        weight_keys=WEIGHT_KEYS, run_keys=RUN_KEYS,
                        weights_only=weights_only)
    agent.net.load_state_dict(state["net"], agent.device)
    # No act can be in flight: `observe` consumes the pending forward pass in
    # the same loop iteration that produced it.
    agent._pending = None
    if not load_run:
        # Weights for a score or a fresh start: no rollout to continue.
        agent.buffer.reset()
        if agent._recurrent is not None:
            agent._recurrent.zero()
        return
    agent.optimizer.load_state_dict(state["optimizer"], agent.device)
    agent.frame_count = int(state["frame_count"])
    agent.update_count = int(state["update_count"])
    agent._rng.bit_generator.state = state["rng_state"]
    agent._autoreset.load_state_dict(state["autoreset"])
    load_rollout_state(agent, state["rollout"])
