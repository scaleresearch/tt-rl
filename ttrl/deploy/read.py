"""Reading a policy artifact off disk, with its version gate.

Split from `ttrl.deploy.artifact`, which is the deployed FORWARD: this is the
FILE. One reader of the layout, so `ttrl.load_policy` building the numpy
forward and `ttrl.deploy.graph` building the ONNX graph do not each have an
idea of what the members are called, and one place that decides whether a
file this build can read was handed to it.

Imports `json` and `numpy` and nothing else, as the forward does: a robot's
host reads the file with what it already has.
"""

from __future__ import annotations

import json

import numpy as np

from ttrl.deploy.artifact import DeployedPolicy

__all__ = ["ARTIFACT_FORMAT", "META_KEY", "load_policy", "read_artifact"]

#: An opaque marker a reader compares for equality, so a file this build did
#: not write is refused rather than read as an array it cannot place. Not a
#: number: an ordered version invites a `<=` and a bump, which is how the
#: checkpoint format grew a migration ladder that had to be deleted. The
#: artifact format is pre-release and carries no migrations, so replacing this
#: string wholesale is the only way to change it.
ARTIFACT_FORMAT = "ttrl-policy-9c2e57"

#: The JSON member. A uint8 array, like a checkpoint's, so the file stays a
#: plain npz with no pickle in it and loading one cannot execute code.
META_KEY = "__meta__"


def read_artifact(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """An artifact's `(meta, arrays)`, with the version gate applied.

    The one reader of the file's layout. `load_policy` builds the numpy
    forward from it and `ttrl.deploy.graph` builds the ONNX graph from it, so
    neither has its own idea of what the members are called.
    """
    with np.load(path, allow_pickle=False) as npz:
        if META_KEY not in npz.files:
            raise ValueError(
                f"{path!r} has no {META_KEY!r} member, so it is not a ttrl "
                f"policy artifact. Write one with ttrl.export_policy."
            )
        meta = json.loads(npz[META_KEY].tobytes().decode("utf-8"))
        arrays = {k: npz[k] for k in npz.files if k != META_KEY}
    if "artifact_format" not in meta and "format" in meta:
        # A checkpoint is an npz with a `__meta__` member too, and handing one
        # to a reader of artifacts is the obvious mistake to make. Saying
        # which file this is beats reporting its artifact version as None.
        raise ValueError(
            f"{path!r} is a ttrl CHECKPOINT (format marker "
            f"{meta['format']!r}) rather than a policy artifact. Turn it into "
            f"one with ttrl.export_policy(checkpoint_path, artifact_path) and "
            f"read that."
        )
    marker = meta.get("artifact_format")
    if marker != ARTIFACT_FORMAT:
        raise ValueError(
            f"{path!r} does not carry the policy artifact marker "
            f"{ARTIFACT_FORMAT!r} this build writes (it carries {marker!r}). "
            f"The artifact format is pre-release: it carries no migrations and "
            f"reads only what this build wrote. Export it again from its "
            f"checkpoint with ttrl.export_policy(checkpoint_path, path)."
        )
    return meta, arrays


def load_policy(path: str) -> DeployedPolicy:
    """Read an artifact `ttrl.export_policy` wrote."""
    return DeployedPolicy(*read_artifact(path))
