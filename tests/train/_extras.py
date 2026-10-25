"""What a core install does NOT have, read off `pyproject.toml`."""

from __future__ import annotations

import pathlib
import re
import tomllib

__all__ = ["EVERYTHING", "EXTRAS", "IMPORT_NAMES", "SUITES", "SUITE_EXTRAS"]

#: The import name each optional distribution is reached by, because the two
#: do not follow from each other (`ale-py` imports as `ale_py`,
#: `opencv-python-headless` as `cv2`). Checked against the extras by
#: `test_every_optional_distribution_has_an_import_name`.
IMPORT_NAMES = {
    "gymnasium": "gymnasium", "ale-py": "ale_py", "mujoco": "mujoco",
    "opencv-python-headless": "cv2", "tensorboard": "tensorboard",
    "wandb": "wandb", "matplotlib": "matplotlib", "onnx": "onnx",
}

#: Which extras hold an environment suite.
SUITE_EXTRAS = ("atari", "mujoco")

#: `dev` is every extra at once plus the test tooling, so blocking what it
#: pulls in would block pytest itself.
_NOT_AN_EXTRA = ("dev",)


def _extras() -> dict[str, tuple[str, ...]]:
    """`extra -> its distributions`."""
    root = pathlib.Path(__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as fh:
        table = tomllib.load(fh)["project"]["optional-dependencies"]
    return {
        extra: tuple(re.split(r"[><=\[]", req, maxsplit=1)[0].strip()
                     for req in reqs)
        for extra, reqs in table.items() if extra not in _NOT_AN_EXTRA
    }


EXTRAS = _extras()

#: Everything a core install must import without. ttnn is in it because a
#: launcher has to scope the visible chips before the first `import ttnn`;
#: torch because it ships in the dev extra and nothing in `ttrl/` may reach
#: it, this library computing on ttnn.
EVERYTHING = ("ttnn", "torch") + tuple(sorted(
    {IMPORT_NAMES[dist] for dists in EXTRAS.values() for dist in dists}))

#: The optional environment suites alone.
SUITES = tuple(sorted({IMPORT_NAMES[dist] for extra in SUITE_EXTRAS
                       for dist in EXTRAS[extra]}))
