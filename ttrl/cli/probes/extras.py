"""Whether each optional extra actually imports.

Imports them for real rather than locating them: an extra that is installed
but broken (a cv2 missing a system library, say) is exactly the case worth
catching, and find_spec would call it present.
"""

from __future__ import annotations

import importlib
from typing import Iterator

from ttrl.cli.probes.versions import providing_distributions
from ttrl.cli.report import OK, WARN, Line

#: extra name -> the modules installing it must make importable.
EXTRAS = {
    "atari": ("gymnasium", "ale_py", "cv2"),
    "mujoco": ("gymnasium", "mujoco"),
    "logging": ("tensorboard",),
}


def _install_hint(extra: str) -> str:
    """How to install an extra, worded for wherever doctor is running.

    The project may be the user's checkout or a dependency of some other
    project, so `-e '.[extra]'` is wrong advice about half the time.
    """
    providers = providing_distributions()
    if not providers:
        return (f"install the {extra} extra into whichever environment provides "
                "ttrl; see the versions section above, which could not identify "
                "an installed distribution")
    name = providers[0]
    return (f"install it where you install {name}: "
            f"`uv pip install '{name}[{extra}]'`, or "
            f"`uv pip install -e '.[{extra}]'` inside a {name} checkout")


def probe() -> Iterator[Line]:
    for extra, modules in EXTRAS.items():
        broken: list[str] = []
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as exc:
                broken.append(f"{name} ({type(exc).__name__}: {exc})")
        if broken:
            # Not essential: an extra is optional by definition, and doctor
            # must not fail a box that simply does not want MuJoCo.
            yield Line(
                WARN, f"extra [{extra}]", "; ".join(broken), essential=False,
                action=f"only needed for {extra}. {_install_hint(extra)}.",
            )
        else:
            yield Line(OK, f"extra [{extra}]", f"{', '.join(modules)} import")
