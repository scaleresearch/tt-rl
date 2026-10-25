"""Multi-chip training of one model: N independent actor-learners on their own
single-chip devices, with periodic exact parameter averaging.

Each chip runs the unchanged single-chip engine: its own optimizer and its own
traces. Every `sync_every` updates `ParamSync.average` reduces the parameter
buffers in place, at fixed addresses, so the traces stay valid.
"""

from ttrl.engine.mesh.devices import open_independent_devices_for_chips
from ttrl.engine.mesh.open import (
    chips_of_open_mesh,
    open_mesh_for_chips,
    replica_axis,
)
from ttrl.engine.mesh.resume import reset_optimizer_state, resume_replicas
from ttrl.engine.mesh.sync import (
    ParamSync,
    host_all_reduce_sum,
    param_schema,
    replica_schema,
)
from ttrl.engine.mesh.topology import (
    ParamSyncError,
    local_ids_for,
    require_exact_visible_scope,
    resolve_chips,
)

__all__ = [
    "ParamSync",
    "ParamSyncError",
    "chips_of_open_mesh",
    "host_all_reduce_sum",
    "local_ids_for",
    "open_independent_devices_for_chips",
    "open_mesh_for_chips",
    "param_schema",
    "replica_axis",
    "replica_schema",
    "require_exact_visible_scope",
    "reset_optimizer_state",
    "resolve_chips",
    "resume_replicas",
]
