"""Sinks that ship a run's scalars to a viewer: TensorBoard and wandb.

Separate from `ttrl.train.logging` because of what they cost: each needs a
package this library does not install, so each is an extra. Exported from
`ttrl.train.logging` all the same, since a caller choosing a sink chooses
among all of them.

EACH ONE DECLARES ITS DEPENDENCY AND REFUSES WITHOUT IT. The import is inside
the constructor, so `import ttrl` costs nothing whether the package is
installed or not, and the refusal names the extra that provides it: a caller
who asked for TensorBoard and silently got nothing has no way to find out.
Nothing here degrades to a no-op sink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["REQUIRES", "TensorBoardLogger", "WandbLogger"]

#: Each sink here, the package it imports, and the extra that installs it.
#: ONE source: the refusal message is built from this, so the package a
#: constructor imports and the name its message tells a user to install cannot
#: drift, and a test can enumerate the sinks that have a dependency to be
#: missing rather than being handed a list.
REQUIRES = {
    "TensorBoardLogger": ("tensorboard", "logging"),
    "WandbLogger": ("wandb", "tracker"),
}


def _missing(sink: str, exc: ModuleNotFoundError) -> ModuleNotFoundError:
    """The refusal: what is missing, how to install it, and the way around it."""
    package, extra = REQUIRES[sink]
    return ModuleNotFoundError(
        f"{sink} needs the {package} package, which is not installed "
        f"({exc.name}). Install it with: pip install 'tt-rl[{extra}]', or log "
        f"somewhere else by passing a different Logger."
    )


class TensorBoardLogger:
    """Scalars into a TensorBoard event file under `<log_dir>/tb`.

    Uses the standalone `tensorboard` package's writer, which needs no torch.
    """

    def __init__(self, log_dir: str | Path) -> None:
        try:
            # Optional dependency, imported on first use; the refusal below names the extra.
            from tensorboard.summary import Writer
        except ModuleNotFoundError as exc:
            raise _missing("TensorBoardLogger", exc) from exc
        self.log_dir = Path(log_dir)
        self._writer = Writer(str(self.log_dir / "tb"))

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._writer.add_scalar(key, float(value), step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

class WandbLogger:
    """Scalars to a Weights & Biases run.

    The same shape as `TensorBoardLogger`: the dependency is optional, its
    absence raises here at construction naming the extra, and nothing is
    imported at module scope, so `import ttrl` is unaffected by whether wandb
    is installed.

    `config` is the run configuration the tracker should show beside the
    curves. Pass `ttrl.train.config.config_to_dict(config)`; this sink does
    not reach for it itself, because a sink that reads a `TrainConfig` would
    be a logging module that knows the config schema.

    This sink OWNS the run it starts and finishes it on `close`. A caller who
    has already called `wandb.init` themselves passes the run as `run` and
    this sink logs into it and leaves it open, because finishing a run it did
    not start would end a process-wide object out from under its owner.
    """

    def __init__(
        self,
        project: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        run=None,
        **init_kwargs: Any,
    ) -> None:
        try:
            # Optional dependency, imported on first use; the refusal below names the extra.
            import wandb
        except ModuleNotFoundError as exc:
            raise _missing("WandbLogger", exc) from exc
        self._owns_run = run is None
        self._run = wandb.init(
            project=project, name=name, config=config, **init_kwargs
        ) if run is None else run

    def log(self, metrics: dict[str, Any], step: int) -> None:
        scalars = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if scalars:
            self._run.log(scalars, step=step)

    def flush(self) -> None:
        # A tracker run has no flush of its own: the client batches and ships
        # on its own schedule, and there is no call that forces it. Saying so
        # here is the honest answer to a protocol method this sink cannot
        # implement, and silently doing nothing elsewhere would be the lie.
        return None

    def close(self) -> None:
        if self._owns_run:
            self._run.finish()
