"""ttrl.engine: the Tenstorrent compute substrate.

Lazy by design. Importing this package imports no ttnn, opens no device and
reads no environment, so that a launcher can call scope_to_chip before ttnn
initialises and reads TT_VISIBLE_DEVICES.
"""

from __future__ import annotations

import importlib
from typing import Any

from ttrl.engine.config import DEFAULT, EngineConfig

_LAZY = {
    "ActingCarry": "carry",
    "ChipLivenessCheckFailed": "device",
    "ChipScopeError": "device",
    "ChipStatus": "device",
    "DeviceProfile": "device",
    "NoDeviceAvailable": "device",
    "acquire": "device",
    "chip_env": "device",
    "enumerate_chips": "device",
    "local_index": "device",
    "physical_chip_id": "device",
    "physical_chips": "device",
    "purge_device_caches": "device",
    "register_device_cache": "device",
    "scope_to_chip": "device",
    "scope_to_chips": "device",
    "status": "device",
    "survey": "device",
    "visible_chip_ids": "device",
    "MASTER_DTYPE": "tensor",
    "StaticArena": "tensor",
    "copy_in_place": "tensor",
    "copy_tensor_in_place": "tensor",
    "to_device": "tensor",
    "to_host": "tensor",
    "restore_in_place": "tensor",
    "Tensor": "autograd",
    "backward": "autograd",
    "no_grad": "autograd",
    "Adam": "optim",
    "Optimizer": "optim",
    "adam_fused_scalars": "optim",
    "TrainStep": "step",
    "read_metrics": "step",
    "CapturedTrace": "trace",
    "TraceCaptureError": "trace",
    "capture": "trace",
    "ParamSync": "mesh",
    "ParamSyncError": "mesh",
    "chips_of_open_mesh": "mesh",
    "host_all_reduce_sum": "mesh",
    "open_independent_devices_for_chips": "mesh",
    "open_mesh_for_chips": "mesh",
    "replica_axis": "mesh",
    "reset_optimizer_state": "mesh",
    "resolve_chips": "mesh",
    "resume_replicas": "mesh",
}

#: Op names are resolved from ttrl.engine.ops on demand rather than listed
#: twice; the ops package owns that list.
_OPS = "ttrl.engine.ops"

__all__ = [
    "DEFAULT",
    "EngineConfig",
    *sorted(_LAZY),
    "add", "div", "mul", "sub",
    "abs_", "exp", "log", "neg", "relu", "sigmoid", "sqrt", "square", "tanh",
    "clamp", "maximum", "minimum",
    "matmul",
    "argmax_dim_detached", "max_dim_detached", "mean_all", "mean_dim",
    "sum_all", "sum_dim",
    "concat", "reshape", "split",
    "apply_done_mask", "entropy", "gather_onehot", "huber", "log_softmax",
    "masks", "non_finite_check", "one_hot_device", "one_hot_gather",
    "one_hot_host",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is not None:
        return getattr(importlib.import_module(f"ttrl.engine.{module}"), name)
    if name in __all__:
        return getattr(importlib.import_module(_OPS), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
