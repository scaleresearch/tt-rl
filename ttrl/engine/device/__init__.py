"""Device acquisition, visibility scoping, liveness, and the device profile.

Importing this package does not import ttnn and reads no environment: a
launcher must be able to call scope_to_chip before the first import of ttnn.
"""

from ttrl.engine.device.acquire import (
    acquire,
    purge_device_caches,
    register_device_cache,
)
from ttrl.engine.device.liveness import (
    DEFAULT_WAIT_SECONDS,
    POLL_SECONDS,
    STALE_FD_NOTE,
    ChipLivenessCheckFailed,
    ChipStatus,
    NoDeviceAvailable,
    describe_holders,
    enumerate_chips,
    physical_chips,
    status,
    survey,
)
from ttrl.engine.device.profile import DeviceProfile
from ttrl.engine.device.scope import (
    MESH_GRAPH_DESC_ENV,
    MESH_GRAPH_DESC_NAME,
    VISIBLE_DEVICES_ENV,
    ChipScopeError,
    chip_env,
    chips_env,
    device_already_open,
    local_index,
    mesh_graph_descriptor_path,
    physical_chip_id,
    scope_to_chip,
    scope_to_chips,
    visible_chip_ids,
)

__all__ = [
    "ChipLivenessCheckFailed",
    "ChipScopeError",
    "ChipStatus",
    "DeviceProfile",
    "MESH_GRAPH_DESC_ENV",
    "MESH_GRAPH_DESC_NAME",
    "DEFAULT_WAIT_SECONDS",
    "NoDeviceAvailable",
    "POLL_SECONDS",
    "STALE_FD_NOTE",
    "VISIBLE_DEVICES_ENV",
    "acquire",
    "chip_env",
    "chips_env",
    "describe_holders",
    "device_already_open",
    "enumerate_chips",
    "local_index",
    "mesh_graph_descriptor_path",
    "physical_chip_id",
    "physical_chips",
    "purge_device_caches",
    "register_device_cache",
    "scope_to_chip",
    "scope_to_chips",
    "status",
    "survey",
    "visible_chip_ids",
]
