"""Prose lint: a comment or docstring must not name code that is not there.

The failure this exists for. Three docstrings were found in one morning
describing a world that no longer existed: a NOTE about a render path that had
been rewritten, a KNOWN GAP about torch that had been closed, and a claim that
`train.py` calls `scope_to_chip` when it no longer did. Prose rots silently,
because nothing executes it.

WHAT IS MECHANICAL, AND WHAT IS NOT. A sentence's meaning is not checkable.
Its *references* are. Every prose block in this repository is scanned for two
kinds of reference that must resolve, and nothing else is claimed:

  paths      a repo-relative path like `ttrl/engine/conv/config.py` or
             `tests/tools/test_shipped_configs.py`. The file must exist.
  dotted     a dotted name rooted at `ttrl`, `tests` or `tools`, like
             `ttrl.engine.step.MODES` or `ttrl.envs.Env`. It must resolve to a
             module in the tree, or to a module plus a name that module's
             source actually binds.

Resolution is by source inspection, never by import: the lint must run on a
host with no chip, and importing `ttrl.engine` would import ttnn.

A reference inside a string that is not a docstring is not prose and is not
checked, because it is code. Absolute paths (`/dev/tenstorrent/0`), globs and
placeholders (`configs/*.py`, `runs/<name>/metrics.jsonl`) are skipped: they
name no single file.

Run as a script (`python -m tools.lint_prose`) or import `violations(root)`
from a test.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Violation", "prose_blocks", "references", "violations", "check", "main"]

#: Top-level packages a dotted reference may be rooted at. These are exactly
#: the importable trees this repository owns; anything else in a dotted name
#: (`numpy.ndarray`, `ttnn.Tensor`) belongs to a dependency whose surface this
#: lint has no business asserting.
ROOTS = ("ttrl", "tests", "tools")

#: Suffixes a path-shaped reference may carry. A reference to a file kind not
#: listed here is not checked rather than silently passed.
PATH_SUFFIXES = ("py", "md", "sh", "json", "yml", "yaml", "txt", "csv", "jsonl",
                 "npz", "toml", "lock")

_PATH = re.compile(
    r"(?<![\w./-])(([A-Za-z_][\w-]*)/(?:[\w.-]+/)*[\w.-]+\.(?:"
    + "|".join(PATH_SUFFIXES) + r"))(?![\w/-])"
)
_DOTTED = re.compile(
    r"(?<![\w.])((?:" + "|".join(ROOTS) + r")(?:\.\w+)+)(?![-\w])"
)


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    reference: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: `{self.reference}` {self.reason}"


def prose_blocks(text: str) -> list[tuple[int, str]]:
    """Every docstring and every `#` comment in `text`, as (lineno, prose).

    Both kinds are collected, because both rot. Comments come from `tokenize`
    rather than a regex so that a `#` inside a string literal is not mistaken
    for one, and docstrings come from the AST so that a module-level string
    used as data is not mistaken for prose.
    """
    blocks: list[tuple[int, str]] = []
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        # The docstring node is the first statement; its own lineno is where
        # the quotes open, which is what a reader needs to find it.
        first = node.body[0]
        blocks.append((getattr(first, "lineno", 1), doc))
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            blocks.append((token.start[0], token.string))
    return sorted(blocks)


def references(prose: str) -> tuple[list[str], list[str]]:
    """(paths, dotted names) referred to by one prose block.

    Each path comes with its own first segment, which decides whether the
    reference is about this repository at all.
    """
    paths = [(m.group(1), m.group(2)) for m in _PATH.finditer(prose)]
    dotted = [m.group(1) for m in _DOTTED.finditer(prose)]
    # A dotted name whose tail is a file suffix is a path written with dots by
    # the path regex's standards; the path check already covered it.
    return paths, [d for d in dotted if d.rsplit(".", 1)[-1] not in PATH_SUFFIXES]


def _module_path(root: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")
    base = root.joinpath(*parts)
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    return None


def _bound_names(path: Path) -> set[str]:
    """Names `path`'s source binds at module or class scope, without importing.

    Class attributes count: `ttrl.spaces.ObsSpec.vector` is a reference a
    reader will follow, and refusing to resolve it would make the lint noisy
    enough to be switched off.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Lazily exported names exist only as strings: `ttrl/__init__.py`
            # and `ttrl/engine/__init__.py` both resolve their public surface
            # through a name-to-module table, and `__all__` is a tuple of
            # strings. A string literal is therefore evidence that the module
            # binds the name, and refusing it would flag most of the public
            # API as missing.
            names.add(node.value)
    return names


def _resolve_dotted(root: Path, dotted: str) -> str | None:
    """None if `dotted` resolves, else why it does not."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        module = _module_path(root, ".".join(parts[:cut]))
        if module is None:
            continue
        tail = parts[cut:]
        if not tail:
            return None
        if tail[0] in _bound_names(module):
            return None
        return (f"is not a name {module.relative_to(root)} binds "
                f"(looked for {tail[0]!r})")
    return "names no module in this tree"


def _resolve_path(root: Path, reference: str, first: str) -> str | None:
    """None if `reference` resolves, or if its first segment is not a
    directory of this repository, in which case it names something outside the
    tree and this lint cannot adjudicate it."""
    if not (root / first).is_dir():
        return None
    if (root / reference).exists():
        return None
    return "names no file in this tree"


def violations(root: Path, *, trees: tuple[str, ...] = ROOTS) -> list[Violation]:
    """Every unresolvable prose reference under `root`'s `trees`."""
    found: list[Violation] = []
    for tree in trees:
        for path in sorted((root / tree).rglob("*.py")):
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            try:
                blocks = prose_blocks(text)
            except (SyntaxError, tokenize.TokenError):
                continue
            for lineno, prose in blocks:
                paths, dotted = references(prose)
                for reference, first in paths:
                    reason = _resolve_path(root, reference, first)
                    if reason:
                        found.append(Violation(path, lineno, reference, reason))
                for reference in dotted:
                    reason = _resolve_dotted(root, reference)
                    if reason:
                        found.append(Violation(path, lineno, reference, reason))
    return found


def check(root: Path) -> bool:
    vs = violations(root)
    for v in vs:
        print(v)
    return not vs


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent
    ok = check(root)
    if ok:
        print("prose lint: clean")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
