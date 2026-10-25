"""The checkpoint format and a run's history of them.

`tree.py` is the type-tagged value tree the JSON meta is made of, and owns
the on-disk field-name contract. `file.py` is the npz itself: levels, atomic
write, format marker, and what every member means. `read.py` reads one back.
`history.py` is the frame-keyed naming, the stable latest symlink and the
retention bound. `best.py` is the one checkpoint the retention bound does not
apply to, the best-scoring one.
"""

from __future__ import annotations

from ttrl.train.checkpoint.best import (
    BEST,
    MODES,
    BestScore,
    best_path,
    describe,
)
from ttrl.train.checkpoint.file import FORMAT, L0, L1, LEVELS, save_checkpoint
from ttrl.train.checkpoint.read import load_checkpoint, read_meta
from ttrl.train.checkpoint.history import (
    LATEST,
    history,
    history_path,
    point_at_latest,
    prune,
)
from ttrl.train.checkpoint.tree import CheckpointFieldError
from ttrl.train.checkpoint.types import register_checkpoint_type

__all__ = [
    "BEST",
    "BestScore",
    "CheckpointFieldError",
    "FORMAT",
    "L0",
    "L1",
    "LATEST",
    "LEVELS",
    "MODES",
    "best_path",
    "describe",
    "history",
    "history_path",
    "load_checkpoint",
    "point_at_latest",
    "prune",
    "read_meta",
    "register_checkpoint_type",
    "save_checkpoint",
]
