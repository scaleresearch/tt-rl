"""One environment, built to hand frames over.

Separate from `ttrl.train.record` because constructing a renderable env and
rolling a policy into an mp4 fail for unrelated reasons: this module's refusals
are about a builder that cannot be asked for frames or a machine that cannot
open a context, and none of them involve a checkpoint.
"""

from __future__ import annotations

import inspect
import re

from ttrl.envs import supports_render
from ttrl.envs.registry import ENVS
from ttrl.train.config.schema import EnvConfig

__all__ = ["render_env"]


#: What a windowing failure inside a C render library looks like from Python.
_NEEDS_A_GL_BACKEND = re.compile(r"gladLoadGL|GLFW|OpenGL|EGL|display",
                                 re.IGNORECASE)


def render_env(env_config: EnvConfig):
    """One `Env` from `env_config`, built to hand over frames.

    Refuses before construction when the builder cannot take `render_mode`,
    and after it when the stack cannot render anyway.
    """

    builder = ENVS.resolve(env_config.name)
    parameters = inspect.signature(builder).parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in parameters.values())
    if "render_mode" not in parameters and not takes_kwargs:
        raise ValueError(
            f"env {env_config.name!r} ({builder.__module__}.{builder.__qualname__}) "
            f"takes no 'render_mode' keyword, so nothing can ask it for frames. "
            f"A recordable builder accepts render_mode='rgb_array'; see "
            f"ttrl.envs.render.Renderable.")
    try:
        env = builder(**{**env_config.kwargs, "render_mode": "rgb_array"})
    except BaseException as exc:
        # MuJoCo's offscreen context picks GLFW by default, which needs a
        # display. On a headless machine it dies inside the C library naming
        # only the symbol it could not load, which says nothing about the one
        # environment variable that fixes it.
        if _NEEDS_A_GL_BACKEND.search(str(exc)) is None:
            raise
        raise RuntimeError(
            f"env {env_config.name!r} could not open a render context: "
            f"{exc}. This machine has no display, so the offscreen renderer "
            f"has to be named: MUJOCO_GL=egl, set before the process starts."
        ) from exc
    if not supports_render(env):
        env.close()
        raise ValueError(
            f"env {env_config.name!r} was built with render_mode='rgb_array' and "
            f"still reports that it cannot render (supports_render is False). "
            f"Its builder has to pass the render mode down, and the stack "
            f"has to implement "
            f"ttrl.envs.render.Renderable: can_render() and render().")
    return env


