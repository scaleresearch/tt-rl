"""Can ttnn be used on this box at all, asked without importing it."""

from __future__ import annotations

import importlib.util


def unavailable() -> str | None:
    """Why ttnn cannot be used here, or None if it can be located.

    Only find_spec is asked, because locating ttnn must not import it: a
    launcher may still have to set the environment ttnn reads at import time.
    An import hook that raises instead of returning None is itself the answer
    to the question, so its exception is reported rather than propagated.
    """
    try:
        if importlib.util.find_spec("ttnn") is None:
            return "no `ttnn` on sys.path"
    except Exception as exc:
        return f"the import system refuses to locate it: {type(exc).__name__}: {exc}"
    return None
