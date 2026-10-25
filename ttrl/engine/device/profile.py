"""DeviceProfile: runtime-probed facts about an opened device.

The only place allowed to know arch names or ttnn compute-kernel config class
names. Everything above asks the profile. ttnn is imported inside the methods
so that importing this package stays ttnn-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from ttrl.engine.config import EngineConfig


@dataclass(frozen=True)
class DeviceProfile:
    arch: str
    tile_shape: tuple[int, int]
    #: compute_with_storage_grid_size() as (x, y).
    grid: tuple[int, int]
    dram_grid: tuple[int, int]
    num_devices: int
    l1_small_size: int
    trace_region_size: int

    def compute_kernel_config(
        self,
        *,
        math_fidelity=None,
        math_approx_mode: bool = True,
        fp32_dest_acc_en: bool = False,
        packer_l1_acc: bool = False,
        dst_full_sync_en: bool = False,
    ):
        # Deferred: this module loads before chip scoping, and ttnn must not be
        # imported until TT_VISIBLE_DEVICES is set.
        import ttnn

        return ttnn.init_device_compute_kernel_config(
            self._arch_enum,
            math_fidelity=ttnn.MathFidelity.HiFi4 if math_fidelity is None else math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
            dst_full_sync_en=dst_full_sync_en,
        )

    @property
    def _arch_enum(self):
        # Deferred: this module loads before chip scoping, and ttnn must not be
        # imported until TT_VISIBLE_DEVICES is set.
        import ttnn

        return {
            "blackhole": ttnn.Arch.BLACKHOLE,
            "wormhole_b0": ttnn.Arch.WORMHOLE_B0,
            "quasar": ttnn.Arch.QUASAR,
        }[self.arch]


def probe_profile(device, cfg: EngineConfig) -> DeviceProfile:
    # Deferred: this module loads before chip scoping, and ttnn must not be
    # imported until TT_VISIBLE_DEVICES is set.
    import ttnn

    def _xy(coord) -> tuple[int, int]:
        return (coord.x, coord.y)

    return DeviceProfile(
        arch=ttnn.get_arch_name(),
        tile_shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        grid=_xy(device.compute_with_storage_grid_size()),
        dram_grid=_xy(device.dram_grid_size()),
        num_devices=device.get_num_devices(),
        l1_small_size=cfg.l1_small_size,
        trace_region_size=cfg.trace_region_size,
    )
