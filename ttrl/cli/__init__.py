"""The `ttrl` command: a thin argparse shell over ttrl's public API.

Two subcommands and nothing more. Everything each one does is something a
caller can do by importing ttrl, because this package adds no capability of
its own; if it ever needs to reach past the public surface, the public surface
is what should change.

This is the only place in the library allowed to print or to exit a process.
"""

from __future__ import annotations

import argparse

from ttrl import __version__
from ttrl.cli import doctor, train

_COMMANDS = (
    ("train", "train a run from a config file", train),
    ("doctor", "check whether this machine can train, and what to fix", doctor),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttrl", description="Reinforcement learning on Tenstorrent.",
    )
    parser.add_argument("--version", action="version", version=f"ttrl {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, help_text, module in _COMMANDS:
        sub = subparsers.add_parser(
            name, help=help_text, description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(sub)
        sub.set_defaults(_run=module.run)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return args._run(args)
