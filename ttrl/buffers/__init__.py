"""Host-side experience storage: one replay buffer and one rollout buffer.

Internal to the algorithms. An agent owns its own experience storage by
design (`ttrl.algos.base`), so no contract asks an application author for a
buffer and nothing here is part of the `ttrl` surface: `ttrl.buffers` is not
exported from the top level, and these classes are free to change shape with
the algorithms that use them.

Everything in this package is numpy over host memory. It imports no ttnn, opens
no device and reads no environment, so the sampling semantics that are easiest
to get silently wrong are fully testable on a machine with no Tenstorrent
hardware at all.
"""

from ttrl.buffers.replay import ReplayBuffer
from ttrl.buffers.rollout import RolloutBuffer
from ttrl.buffers.types import (ReplayBatch, RolloutMinibatch,
                                RolloutSequenceBatch)

__all__ = ["ReplayBatch", "ReplayBuffer", "RolloutBuffer", "RolloutMinibatch",
           "RolloutSequenceBatch"]
