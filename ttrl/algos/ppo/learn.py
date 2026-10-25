"""PPO's learning half: one rollout turned into gradient steps.

Separated from the agent for the reason `ttrl.algos.sac.step` is: the agent
owns the acting pair, the rollout buffer and the boundary fan-out, and this
owns the shape of one update. The split is along the rollout: everything here
runs after `should_update` says the buffer is full and before it is reset, and
nothing here touches `act`, `observe` or the `EpisodeBoundaries` seam those
two share.

These are functions over an agent rather than methods on one because the
update reads most of what a `PPO` holds -- the config, the buffer, the
optimizer, the acting net and, when there is one, the recurrent runtime -- and
a class carrying the same five references would be a second place to keep them
in step. `PPO.update` and `PPO._minibatch_step` are the one-line methods that
call in here.

Both paths run host-side: `ttrl.algos.ppo.loss` builds the objective and the
optimizer steps outside any captured region, so the metric readbacks below are
legal where SAC's would not be.
"""

from __future__ import annotations

import numpy as np

from ttrl.algos.grads import clip_grad_norm
from ttrl.algos.ppo.loss import objective
from ttrl.engine.autograd import no_grad
from ttrl.engine.ops import mean_all

__all__ = ["last_values", "minibatch_step", "update"]


def last_values(agent) -> np.ndarray:
    """V(s) for the observation after the rollout's last stored step."""
    with no_grad():
        if agent._recurrent is None:
            _policy_out, value = agent._obs(agent._last_next_obs)
        else:
            value = agent._recurrent.bootstrap(
                agent._obs, agent.obs_spec.observation(agent._last_next_obs),
                lanes=agent.n_envs)
    return value.numpy().reshape(-1)[: agent.n_envs].astype(np.float32)


def update(agent) -> dict[str, float]:
    """Every epoch's minibatches over one full rollout, then reset it."""
    cfg = agent.cfg
    if not agent.buffer.full:
        raise RuntimeError(
            f"PPO.update: the rollout holds fewer than its {cfg.n_steps} "
            f"steps. Gate on should_update, which reports when it is full."
        )
    if cfg.anneal_lr:
        frac = max(0.0, 1.0 - agent.frame_count / cfg.total_frames)
        agent.optimizer.lr = cfg.lr * frac
    agent.buffer.compute_gae(cfg.gamma, cfg.lam, last_values(agent))

    totals: dict[str, float] = {}
    count = 0
    batches = (agent.buffer.minibatches if agent._recurrent is None
               else agent.buffer.sequence_minibatches)
    for minibatch in batches(cfg.minibatch_size, epochs=cfg.epochs,
                             normalize_adv=cfg.normalize_adv):
        for key, value in minibatch_step(agent, minibatch).items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    agent.buffer.reset()
    agent.update_count += 1
    metrics = {key: value / count for key, value in totals.items()}
    metrics["lr"] = agent.optimizer.lr
    metrics["minibatches"] = float(count)
    return metrics


def minibatch_step(agent, batch) -> dict[str, float]:
    """One gradient step. `ttrl.algos.ppo.loss` holds the objective both paths
    share: they differ in how the net is called and how the loss reduces."""
    cfg = agent.cfg
    device = agent.device
    if agent._recurrent is None:
        policy_out, value = agent._obs(batch.obs)
        reduce = mean_all
    else:
        # The batch comes back with every excluded slot's learning signal
        # zeroed; ttrl.algos.ppo.recurrent says why the mask alone is not
        # enough.
        policy_out, value, reduce, batch = agent._recurrent.forward(
            agent._obs, batch, device)
    loss, parts, new_logp = objective(
        agent.policy, policy_out, value, batch, cfg, device, reduce=reduce)
    policy_loss = parts["policy_loss"]
    value_loss = parts["value_loss"]
    entropy_bonus = parts["entropy"]

    agent.optimizer.zero_grad()
    loss.backward()
    norm = clip_grad_norm(list(agent.net.parameters()), cfg.max_grad_norm)
    agent.optimizer.step()

    # Schulman's k3 estimator, mean(exp(r) - 1 - r) with r = new - old:
    # unbiased for KL(old || new) and nonnegative per sample, unlike the
    # naive signed difference, which reads negative on a healthy update.
    logratio = new_logp.numpy().reshape(-1) - batch.old_logp.reshape(-1)
    return {
        "loss": float(loss.numpy().reshape(-1)[0]),
        "policy_loss": float(policy_loss.numpy().reshape(-1)[0]),
        "value_loss": float(value_loss.numpy().reshape(-1)[0]),
        "entropy": float(entropy_bonus.numpy().reshape(-1)[0]),
        "grad_norm": norm,
        "approx_kl": float(np.mean(np.expm1(logratio) - logratio)),
    }
