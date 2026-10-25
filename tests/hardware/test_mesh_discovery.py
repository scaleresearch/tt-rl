"""The open-by-discovery path on real hardware, and the device collective."""

from __future__ import annotations

import numpy as np
import pytest

import ttnn

from ttrl.engine.device import visible_chip_ids
from ttrl.engine.mesh import host_all_reduce_sum, open_mesh_for_chips, replica_axis
from ttrl.engine.mesh.open import ParamSyncError

pytestmark = pytest.mark.tt


def _scoped_chips() -> list[int]:
    visible = visible_chip_ids()
    if visible is None:
        pytest.skip("unscoped: refusing to open every chip on the box")
    return visible


def _require(n: int) -> list[int]:
    chips = _scoped_chips()
    if len(chips) < n:
        pytest.skip(
            f"needs {n} chips; this process is scoped to {chips}. Relaunch with "
            f"TT_VISIBLE_DEVICES naming {n} chips of one board."
        )
    return chips[:n]


def test_mesh_opens_with_no_shape_and_no_descriptor():
    chips = _scoped_chips()
    with open_mesh_for_chips(chips) as mesh:
        assert len(mesh.get_device_ids()) == len(chips)
        # The shape is whatever the cabling is. Asserting a particular one
        # would reintroduce the table this path replaced.
        assert int(np.prod([int(d) for d in mesh.shape])) == len(chips)


def test_a_second_mesh_in_this_process_is_refused():
    chips = _scoped_chips()
    with open_mesh_for_chips(chips):
        with pytest.raises(ParamSyncError, match="already holds a mesh"):
            with open_mesh_for_chips(chips):
                pass


def test_device_all_reduce_matches_the_host_sum_exactly():
    """The gate on replacing `host_all_reduce_sum`."""
    chips = _require(2)
    n = len(chips)
    rng = np.random.default_rng(0)
    per_shard = [rng.standard_normal((32, 128)).astype(np.float32) for _ in range(n)]
    expected = host_all_reduce_sum(per_shard)[0]

    with open_mesh_for_chips(chips) as mesh:
        axis = replica_axis(mesh)
        stacked = np.concatenate(per_shard, axis=0)
        sharded = ttnn.from_numpy(
            stacked,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, dim=0),
        )
        reduced = ttnn.all_reduce(sharded, cluster_axis=axis)
        got = ttnn.to_numpy(
            reduced, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0)
        ).astype(np.float32)

    for i in range(n):
        shard = got[i * expected.shape[0] : (i + 1) * expected.shape[0]]
        np.testing.assert_array_equal(shard, expected)
