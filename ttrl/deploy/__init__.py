"""Taking a trained policy off the training box and onto a robot.

Three names:

    ttrl.export_policy(checkpoint_path, out_path)   # training box
    ttrl.load_policy(artifact_path)                 # robot host
    ttrl.export_onnx(artifact_path, out_path)       # anybody else's host

`artifact.py` is the deployed side and imports numpy and nothing else;
`read.py` is the file it is read out of, numpy and `json` and nothing else.
`export.py` is the training side and reads a checkpoint, so it reaches
`ttrl.spaces` and the checkpoint reader; it is imported only here, which is
why reaching `load_policy` does not drag it in.

`graph.py` is the third side: the same artifact as a standards-compliant ONNX
graph, for a deployment stack that is not this one. It reads the artifact
rather than the checkpoint, so the graph computes the function of the file
the robot already has, and it needs the optional `onnx` extra.
"""

from __future__ import annotations

import importlib

__all__ = ["ARTIFACT_FORMAT", "DeployedPolicy", "export_onnx", "export_policy",
           "load_policy"]

_HOMES = {
    "ttrl.deploy.artifact": ("ACTION_MAPS", "DeployedPolicy"),
    "ttrl.deploy.read": ("ARTIFACT_FORMAT", "META_KEY", "load_policy",
                         "read_artifact"),
    "ttrl.deploy.export": ("export_policy",),
    "ttrl.deploy.graph": ("OPSET", "export_onnx"),
}

_OWNER = {name: module for module, names in _HOMES.items() for name in names}


def __getattr__(name):
    module = _OWNER.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(__all__) | set(globals()))
