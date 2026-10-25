"""Where the action enters Q: the state-action fusion seam, and its defaults.

`ttrl.nets.encoders` builds a tower from one OBSERVATION leaf. Nothing there
has an action. This module is the one place an action is joined to observation
features, which is a different question with a different answer per
observation kind, so it holds both the module that does the joining and the
kind-to-architecture mapping a critic resolves 'auto' through.

The two mappings here and in `ttrl.nets.encoders` are joined by
`ttrl.nets.multi.DEFAULT_ENCODER_BY_ROLE`, which is what `build_encoder`
reads. Each table lives with the thing it describes and the reader picks one
by role.
"""

from __future__ import annotations

from ttrl.engine.nn import Module
from ttrl.engine.ops import concat
from ttrl.nets.encoders import Encoder

__all__ = ["DEFAULT_CRITIC_ENCODER_BY_KIND", "ConcatEncoder"]


#: Encoder name a leaf implies for a CRITIC's tower, the counterpart of
#: `ttrl.nets.encoders.DEFAULT_ENCODER_BY_KIND` for the one role that has an
#: action to fuse. A vector leaf gets 'identity', so `concat(s, a)` is the
#: critic's input and every layer of `critic_hidden` depends on the action;
#: that is the published SAC critic (Haarnoja et al. 2018). An image leaf gets
#: its own conv tower and fuses after it, because there is no spatial extent
#: for an action scalar to occupy in a plane of pixels. The keys are gated
#: against `ObsKind` the same way the observation table's are, so a new kind
#: without a declared critic default cannot pass unnoticed.
DEFAULT_CRITIC_ENCODER_BY_KIND = {"image": "nature_cnn", "vector": "identity"}


class ConcatEncoder(Module):
    """Encode the observation, then append a raw action vector to the features.

    `encoder` is the whole observation's encoder, so the observation it is
    called with is the leaf mapping that encoder takes.

    This is SAC's Q(s, a) input, and WHERE the action enters Q is `encoder`'s
    choice rather than this module's. With a tower of hidden layers ahead of
    it, everything in Q that depends on the action is whatever FOLLOWS this
    concatenation: the state enters the action's nonlinearities as a bias
    shift and cannot multiply it, and the actor's dQ/da flows back through
    that shortened tail. With 'identity' named as the critic's encoder, this
    concatenation is the critic's INPUT and the full depth of `critic_hidden`
    is action-dependent.

    The input-fused form is the default for a VECTOR leaf, by way of
    `DEFAULT_CRITIC_ENCODER_BY_KIND`: it is the published SAC critic. An
    IMAGE leaf still fuses after its conv tower and is under a constraint a
    vector leaf is not: a conv stage takes a plane of pixels, and an action
    scalar has no spatial extent to occupy there, so the concatenation cannot
    move ahead of the tower at all. `identity_encoder` refuses an image spec
    by name for that reason rather than flattening one.

    Both forms are reached by a NAME, resolved from the leaf's declared kind
    or named outright in the config, so a topology is always declared and
    recorded in the checkpoint; nothing here reads a shape and guesses which
    one was meant.
    """

    def __init__(self, encoder: Encoder, action_dim: int):
        self.encoder = encoder
        self.action_dim = int(action_dim)
        self.out_features = encoder.out_features + self.action_dim

    def forward(self, obs, action):
        return concat((self.encoder(obs), action), dim=-1)
