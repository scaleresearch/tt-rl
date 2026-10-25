"""Which network a `PPOConfig` asks for.

One function, because the choice is one line of config and two builders, and
because the refusal below is about a config field rather than about a layer:
the network is built for `minibatch_size`, so a convolution that does not fit
on the chip at that batch is a refusal of that field and is named as one, at
construction, rather than reaching the first backward pass with a chip already
open and a rollout already collected. Every other build failure passes through
unchanged.
"""

from __future__ import annotations

from ttrl.engine.conv import ConvDoesNotFit
from ttrl.nets.build import build_actor_critic
from ttrl.nets.recurrent import build_recurrent_actor_critic

__all__ = ["build_net"]


def build_net(obs_spec, action_spec, device, cfg):
    """The actor-critic `cfg` describes, recurrent when `cfg.recurrent` is set."""
    memory = cfg.recurrent
    try:
        if memory is None:
            return build_actor_critic(
                obs_spec, action_spec, device, batch=cfg.minibatch_size,
                encoder=cfg.encoder, **cfg.encoder_kwargs)
        return build_recurrent_actor_critic(
            obs_spec, action_spec, device, batch=cfg.minibatch_size,
            encoder=cfg.encoder, hidden_size=memory.hidden_size,
            **cfg.encoder_kwargs)
    except ConvDoesNotFit as exc:
        raise ValueError(
            f"PPO: the network does not build at "
            f"minibatch_size={cfg.minibatch_size}: {exc}"
        ) from exc
