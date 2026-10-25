"""`python -m tools.curves`: pool seeds, compare arms, write the figure.

The artefact an experiment is run for is mean plus or minus standard error of
return against frames, one band per arm, pooled over seeds. Every user of
this library was writing that themselves out of `metrics.jsonl`, which is
forty lines of numpy and matplotlib that have nothing to do with their
experiment:

    python -m tools.curves --arm "n_step=1:runs/ab_ns1_s*" \\
                           --arm "n_step=3:runs/ab_ns3_s*" --out runs/nstep.png

WHY `tools/` AND NOT THE LIBRARY. `ttrl`'s command line is three
subcommands, train, eval and doctor, and each is a thin shell over one public
call; a plotting command is neither of those things. Reading a run directory
needs nothing from the library either: `metrics.jsonl` is a documented file
format, one JSON object per line with a `step` key, which is exactly what
makes it worth having. So this is tooling, like the sweep launcher, and
matplotlib stays out of the wheel's dependencies: `pip install
'tt-rl[plots]'` is what the import error names.

The numbers are printed as well as plotted, every time. A band read off a PNG
is not a number anyone can put in a table, and the final mean and standard
error per arm is what a comparison is usually settled by.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Arm", "Band", "band", "curve", "format_table", "main", "parse_arm"]

#: The metric a return curve is made of. Written by every algorithm in the
#: tree at the end of each training episode.
DEFAULT_METRIC = "rollout/episode_return"

#: Frames per bin. An episode return arrives once per episode, at whatever
#: frame the episode happened to end on, so two seeds never share an x value
#: and pooling needs a common grid.
DEFAULT_BIN = 500

#: Matplotlib's default cycle is fine for lines and too strong for bands, so
#: the arms get a fixed, colour-blind-safe order instead.
_COLORS = ("#c0463a", "#2f6f9f", "#4a8a52", "#8a6bbf", "#b4802c")


@dataclass(frozen=True)
class Arm:
    """One arm of a comparison: a label and a glob over run directories."""

    label: str
    pattern: str


@dataclass(frozen=True)
class Band:
    """A pooled arm: frames, mean, standard error, and how many seeds."""

    label: str
    frames: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    seeds: int


def parse_arm(raw: str) -> Arm:
    """Parse one `--arm` value: `label:glob`.

    Split on the LAST colon, so a Windows-style or `ALE/Pong-v5`-style label
    keeps its own punctuation and only the glob is taken off the end.
    """
    label, sep, pattern = raw.rpartition(":")
    if not sep or not label.strip() or not pattern.strip():
        raise ValueError(
            f"arm {raw!r} is not `label:glob`. Pass something like "
            '--arm "n_step=3:runs/ab_ns3_s*".'
        )
    return Arm(label=label.strip(), pattern=pattern.strip())


def metrics_files(pattern: str) -> list[Path]:
    """The `metrics.jsonl` files under the run directories `pattern` matches.

    A pattern naming the files themselves also works, which is what a shell
    that has already expanded one hands over.
    """
    found: list[Path] = []
    for match in sorted(glob.glob(pattern)):
        path = Path(match)
        candidate = path / "metrics.jsonl" if path.is_dir() else path
        if candidate.is_file():
            found.append(candidate)
    return found


def curve(path: Path, *, metric: str = DEFAULT_METRIC, bin_w: int = DEFAULT_BIN):
    """One run's metric, averaged within each `bin_w`-frame bin.

    Records without `metric` are skipped: a JSONL file interleaves rollout,
    training and eval records, and only some of them carry a return.
    """
    steps: list[int] = []
    values: list[float] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if metric in record:
                steps.append(int(record["step"]))
                values.append(float(record[metric]))
    if not steps:
        return np.array([]), np.array([])
    step = np.array(steps)
    value = np.array(values)
    edges = np.arange(0, step.max() + bin_w, bin_w)
    index = np.digitize(step, edges) - 1
    xs, ys = [], []
    for bucket in range(len(edges)):
        mask = index == bucket
        if mask.any():
            xs.append(edges[bucket] + bin_w / 2)
            ys.append(value[mask].mean())
    return np.array(xs), np.array(ys)


def band(
    arm: Arm, *, metric: str = DEFAULT_METRIC, bin_w: int = DEFAULT_BIN
) -> Band | None:
    """Pool every seed of `arm`. None when the pattern matched no data.

    Truncated to the shortest seed, because a tail averaged over one
    surviving seed looks like a result and is one run.
    """
    curves = [
        curve(path, metric=metric, bin_w=bin_w) for path in metrics_files(arm.pattern)
    ]
    curves = [c for c in curves if len(c[0])]
    if not curves:
        return None
    shortest = min(len(c[0]) for c in curves)
    frames = curves[0][0][:shortest]
    stacked = np.stack([c[1][:shortest] for c in curves])
    seeds = len(curves)
    # The sample estimator (ddof=1): with three seeds the population form
    # understates the band by 18%, and three seeds is what an ablation runs.
    # One seed has no spread to estimate, so its band is flat and the legend
    # says how many seeds it is.
    spread = stacked.std(0, ddof=1) if seeds > 1 else np.zeros(shortest)
    return Band(
        label=arm.label,
        frames=frames,
        mean=stacked.mean(0),
        stderr=spread / np.sqrt(seeds),
        seeds=seeds,
    )


def format_table(bands: "list[Band]", *, metric: str) -> str:
    """The final value of each arm, with its seed count and standard error."""
    width = max((len(b.label) for b in bands), default=1)
    lines = [f"{'arm':<{width}}  seeds  frames  final {metric} (mean +/- stderr)"]
    for item in bands:
        lines.append(
            f"{item.label:<{width}}  {item.seeds:>5}  {int(item.frames[-1]):>6}  "
            f"{item.mean[-1]:.2f} +/- {item.stderr[-1]:.2f}"
        )
    return "\n".join(lines)


def plot(bands: "list[Band]", out: Path, *, metric: str, title: str) -> Path:
    """Write the figure. A missing matplotlib raises and names the extra."""
    try:
        # Optional dependency, imported on first use; the refusal below names the extra.
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tools.curves draws with matplotlib, which is not installed "
            f"({exc.name}). Install it with: pip install 'tt-rl[plots]'."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))
    for index, item in enumerate(bands):
        color = _COLORS[index % len(_COLORS)]
        axes.plot(item.frames, item.mean, color=color,
                  label=f"{item.label} ({item.seeds} seeds)")
        axes.fill_between(item.frames, item.mean - item.stderr,
                          item.mean + item.stderr, color=color, alpha=0.2, lw=0)
    axes.set_xlabel("environment frames")
    axes.set_ylabel(metric)
    if title:
        axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.curves",
        description="Pool seeds and compare arms from run directories.",
        epilog='example:\n  python -m tools.curves --out runs/nstep.png '
               '--arm "n_step=1:runs/ab_ns1_s*" --arm "n_step=3:runs/ab_ns3_s*"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm", action="append", default=[], metavar="LABEL:GLOB", required=True,
        help="one arm of the comparison: a label and a glob over the run "
             "directories holding its seeds (repeatable)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="PATH",
        help="where to write the figure",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        help="metrics.jsonl key to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--bin", type=int, default=DEFAULT_BIN, dest="bin_w", metavar="FRAMES",
        help="frames per bin, the grid seeds are pooled on (default: %(default)s)",
    )
    parser.add_argument("--title", default="", help="figure title")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        arms = [parse_arm(raw) for raw in args.arm]
    except ValueError as exc:
        parser.error(str(exc))

    bands: list[Band] = []
    for arm in arms:
        pooled = band(arm, metric=args.metric, bin_w=args.bin_w)
        if pooled is None:
            print(
                f"arm {arm.label!r}: no {args.metric} records under "
                f"{arm.pattern!r}. Check the glob against the run directories, "
                "and --metric against a metrics.jsonl line.",
                file=sys.stderr,
            )
            continue
        bands.append(pooled)
    if not bands:
        print("no arm had any data, so there is nothing to plot", file=sys.stderr)
        return 1
    print(format_table(bands, metric=args.metric))
    print("wrote", plot(bands, args.out, metric=args.metric, title=args.title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
