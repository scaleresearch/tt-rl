"""The registries are actually populated in a fresh interpreter.

A lazy package `__init__` and a registry whose builtins name that PACKAGE are
mutually exclusive: `@ALGOS("dqn")` lives in `dqn/agent.py`, so once
`dqn/__init__.py` stopped importing the agent eagerly, importing the package
registered nothing and `ALGOS.resolve("dqn")` raised "unknown algorithm 'dqn';
registered names are []" -- a message that blames the caller for the name when
the registry is simply empty. The builtins must name the module the decorator
is in, and nothing in-process notices if they stop doing so.

EVERY CASE RUNS IN A SUBPROCESS, and that is the whole test rather than a
detail. This file's own imports would register the builders as a side effect:
`from ttrl.algos.dqn import DQN` resolves through the lazy `__getattr__` and
imports `agent.py`, so an in-process assertion passes whatever the builtins
list says. A fresh interpreter that imports only the registry is the one place
the question can be asked.

A registry whose entries are `"module:attr"` TARGETS has the mirror-image
question, and all four of this package's registries are that shape: the
registration is a call in the registry's own module, so importing it must
leave every name present and every target resolvable, while asking WHICH
names exist must import none of them. Both halves are checked below, the second in a fresh
interpreter for the same reason as the first.

Deliberately separate from `test_algos_no_ttnn.py`: resolving a builder
imports an agent and therefore ttnn, so asserting resolution and asserting
ttnn's absence in one process makes the import order decide the result. That
file holds the membership-without-ttnn case; this one holds resolution.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "ttrl"


def _parsed() -> dict[str, ast.Module]:
    """Every module of the package as a tree, keyed by its import path."""
    out = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(REPO).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = ast.parse(path.read_text())
    return out


def _self_populating(trees) -> dict[str, tuple[str, tuple[str, ...]]]:
    """`registry name -> (module it is built in, the builtins it imports)`.

    A registry with no builtins imports nothing on lookup, so it has no
    builtins question to answer; it is covered by `_target_registries`
    instead.
    """
    found = {}
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "Registry"
                    and isinstance(node.targets[0], ast.Name)):
                continue
            args = list(node.value.args[1:2]) + [
                kw.value for kw in node.value.keywords if kw.arg == "builtins"]
            modules = tuple(
                elt.value for arg in args if isinstance(arg, ast.Tuple)
                for elt in arg.elts if isinstance(elt, ast.Constant))
            if modules:
                found[node.targets[0].id] = (module, modules)
    return found


def _all_registries(trees) -> dict[str, str]:
    """`registry name -> module it is built in`, whatever its shape."""
    found = {}
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "Registry"
                    and isinstance(node.targets[0], ast.Name)):
                found[node.targets[0].id] = module
    return found


def _target_registries(trees) -> dict[str, tuple[str, list[str]]]:
    """`registry name -> (module it is built in, the names registered there)`.

    A `Registry(..., allow_targets=True)` assignment, plus every
    `register("name", "module:attr")` call in that same module, whether
    written through the registry or through a module-level alias of its
    `register`. Read off the source rather than asked of the registry, for
    the reason `_registered_names` gives: this process populates it by the
    same import the subprocess is being tested for.
    """
    found = {}
    for module, tree in trees.items():
        built = [
            node.targets[0].id for node in ast.walk(tree)
            if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Registry"
                and isinstance(node.targets[0], ast.Name)
                and any(kw.arg == "allow_targets" for kw in node.value.keywords))
        ]
        if not built:
            continue
        names = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and len(node.args) == 2):
                continue
            func = node.func
            attr = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if attr != "register":
                continue
            if all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                   for a in node.args) and ":" in node.args[1].value:
                names.append(node.args[0].value)
        for name in built:
            found[name] = (module, sorted(names))
    return found


def _registered_names(trees, registry: str) -> list[str]:
    """The names decorated into `registry` anywhere in the package.

    THE SET IS DERIVED, NOT TYPED. The old list here said "dqn", "ppo", "sac",
    which both misses an algorithm added correctly and fails on the honest
    change that adds one. The decorator sites are the independent source: the
    registry's own `names()` cannot be asked instead, because this file's
    parent process populates it by the same import the subprocess is being
    tested for, which makes the comparison true by construction.

    Read as text rather than imported, because the modules holding the
    decorators reach ttnn and this question needs no chip.
    """
    names = []
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call)
                        and isinstance(deco.func, ast.Name)
                        and deco.func.id == registry
                        and deco.args
                        and isinstance(deco.args[0], ast.Constant)):
                    names.append(deco.args[0].value)
    return sorted(names)


_TREES = _parsed()

#: One case per registry that has builtins to import, with the names its
#: decorators claim. Both halves come off the source.
BUILTIN_CASES = {
    registry: (module, registry, _registered_names(_TREES, registry))
    for registry, (module, _builtins) in sorted(_self_populating(_TREES).items())
}

#: One case per target-string registry, with the names its own module
#: registers. Both halves come off the source, as above.
TARGET_CASES = {
    registry: (module, registry, names)
    for registry, (module, names) in sorted(_target_registries(_TREES).items())
}

CASES = {**BUILTIN_CASES, **TARGET_CASES}


def test_there_are_registries_and_names_to_enumerate():
    """A scan that found nothing would make every case below vacuous.

    Derived on both sides: every registry the package builds must land in one
    bucket or the other, so neither a new registry nor one converted from
    builtins to targets drops out of the cases below unnoticed. No floor on
    `BUILTIN_CASES`: all four take target strings now, and an empty bucket is
    the shape aimed at rather than a failed scan.
    """
    built = _all_registries(_TREES)
    assert built, "no registry was found in ttrl/ at all"
    assert set(CASES) == set(built), (
        f"registries built in ttrl/ are {sorted(built)}, and the ones a case "
        f"below enumerates are {sorted(CASES)}. A registry in neither bucket "
        f"is checked by nothing here: either its builtins tuple or its "
        f"allow_targets=True is missing, or it registers no 'module:attr' "
        f"target the scan can read."
    )
    assert len(TARGET_CASES) >= 2, TARGET_CASES
    for case, (_module, _registry, names) in BUILTIN_CASES.items():
        assert len(names) >= 3, (case, names)
    for case, (_module, _registry, names) in TARGET_CASES.items():
        assert names, (case, names)

_BODY = """
from {module} import {registry} as reg

names = reg.names()
assert names == {expected!r}, f"registry holds {{names}}, expected {expected!r}"
for name in names:
    builder = reg.resolve(name)
    assert callable(builder), f"{{name}} resolves to {{builder!r}}, not a callable"
assert len(reg) == {count}
print("OK")
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", body], cwd=REPO,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("case", list(BUILTIN_CASES))
def test_registry_is_populated_in_a_fresh_interpreter(case):
    module, registry, expected = BUILTIN_CASES[case]
    done = _run(_BODY.format(module=module, registry=registry, expected=expected,
                             count=len(expected)))
    assert done.returncode == 0, (
        f"the {case} registry is not populated by importing {module}:\n"
        f"{done.stdout}\n{done.stderr}"
    )
    assert "OK" in done.stdout


# A subset rather than the whole of `names()`: a registry may also hold
# directly-registered callables, as SAMPLERS holds 'uniform'. What is being
# checked is that every name registered as a target is there, and is still a
# target, after the import alone.
_ENTRY_BODY = """
from {module} import {registry} as reg

# Asked before anything is resolved: each of these answers out of the table,
# and a registry that imported a builder to answer them would reach ttnn
# before the process is scoped to its chip.
listed = reg.names()
assert len(reg) == len(listed)
for name in {expected!r}:
    assert name in listed, f"{{name}} is missing; registry holds {{listed}}"
    assert name in reg, f"{{name}} is registered but not `in` the registry"
    target = reg.entry(name)
    assert isinstance(target, str) and ":" in target, (
        f"{{name}} is not a target string but {{target!r}}"
    )

# Only now, and this is what the target has to be good for: the one query
# that imports.
for name in {expected!r}:
    builder = reg.resolve(name)
    assert callable(builder), f"{{name}} resolves to {{builder!r}}, not a callable"
print("OK")
"""


@pytest.mark.parametrize("case", list(TARGET_CASES))
def test_a_target_registry_lists_its_names_without_resolving_them(case):
    module, registry, expected = TARGET_CASES[case]
    done = _run(_ENTRY_BODY.format(module=module, registry=registry,
                                   expected=expected))
    assert done.returncode == 0, f"{case}:\n{done.stdout}\n{done.stderr}"
    assert "OK" in done.stdout


@pytest.mark.parametrize("case", list(BUILTIN_CASES))
def test_every_registered_builder_lives_where_the_builtins_say(case):
    """The failure mode is a builtins entry naming a module the decorator is
    not in, which leaves the registry empty while every path still imports."""
    module, registry, expected = BUILTIN_CASES[case]
    done = _run(f"""
import sys
from {module} import {registry} as reg

# Read before any lookup: the first one consumes the list.
declared = tuple(reg._builtins)
assert declared, "the registry declares no builtins to import"
# Not vacuous on an empty registry: that is the failure being guarded.
assert reg.names(), f"nothing registered after importing {{declared}}"

for name in reg.names():
    builder = reg.resolve(name)
    home = getattr(builder, "__module__", None)
    assert home in sys.modules, f"{{name}} resolved from an unimported module"
    # The module the builder is defined in must be one the registry imports
    # on its own, or a fresh interpreter finds the registry empty.
    assert any(home == entry or home.startswith(entry + ".")
               for entry in declared), \\
        f"{{name}} is defined in {{home}}, which none of {{declared}} imports"
print("OK")
""")
    assert done.returncode == 0, f"{case}:\n{done.stdout}\n{done.stderr}"
