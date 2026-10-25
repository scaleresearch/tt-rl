"""Dotted-path override parsing and application onto a nested config dict."""

from __future__ import annotations

import ast
from typing import Any


def _coerce_scalar(raw: str) -> Any:
    """Parse an override's right-hand side into a Python value.

    Overrides arrive as bare strings (``"algo_cfg.lr=5e-5"``, ``"device.chip=2"``,
    ``"env.kwargs.game=ALE/Breakout-v5"``, ``"device.chips=[0,1]"``); this recovers
    ``bool``/``None``/``int``/``float``/``list``/``tuple`` where the text plainly
    is one, and otherwise keeps the string (e.g. game ids, log dir paths).

    Bracketed text is a *declared* container: it is parsed with
    :func:`ast.literal_eval` and a syntax error raises here, naming the offending
    text, rather than sliding through as the literal string ``"[0,1]"`` and
    failing much later with a confusing message from whoever consumed it (that
    was a real footgun for ``device.chips``). ``{...}`` is rejected outright:
    nested config is spelled with dotted paths (``device.chips=...``), never
    with an inline mapping.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("{"):
        raise ValueError(
            f"invalid override value {text!r}: inline mappings are not supported; "
            "set nested fields with dotted paths instead (e.g. device.chips=[0,1])"
        )
    if text.startswith("[") or text.startswith("("):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"invalid override value {text!r}: expected a list literal "
                f"such as [0,1] ({exc})"
            ) from None
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"invalid override value {text!r}: expected a list literal such as [0,1]"
            )
        return list(value)
    return text


def _parse_override_list(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}, expected dotted.path=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key or any(not part.strip() for part in key.split(".")):
            raise ValueError(
                f"invalid override {item!r}: empty key segment in {key!r}, "
                "expected dotted.path=value"
            )
        if key in parsed:
            # Two overrides for one field is always a mistake (a typo, or a
            # sweep building argv twice); last-wins silently picks one.
            raise ValueError(f"duplicate override for {key!r}")
        parsed[key] = _coerce_scalar(raw_value.strip())
    return parsed


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | list[str] | None) -> dict[str, Any]:
    """Apply dotted-path overrides (e.g. ``{"algo_cfg.lr": 5e-5}``) onto a nested dict.

    Accepts either a dict of dotted-path -> value, or a list of
    ``"dotted.path=value"`` strings (a common CLI override pattern). Returns a
    new dict; ``data`` is not mutated.

    An unknown key is left for the rebuild in :mod:`ttrl.train.config.loader`
    to refuse, which it does naming the section and the nearest real field.
    That is where the knowledge of what a field is lives, and it covers every
    dataclass section. ``env.kwargs`` is the one place a key is NOT checked,
    and deliberately: it is an environment builder's own keyword set, open by
    definition, so a name this layer does not recognise there is the normal
    case rather than a typo. A keyword the builder does not take is refused by
    the builder, by name, when ``build_env`` constructs the environment.
    """
    if not overrides:
        return data

    if isinstance(overrides, list):
        overrides = _parse_override_list(overrides)

    result = deepcopy_dict(data)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = result
        for depth, part in enumerate(parts[:-1]):
            if part not in node:
                node[part] = {}
            elif not isinstance(node[part], dict):
                # Traversing *through* a leaf: `algo.foo=1` would otherwise
                # silently replace the `algo` string with a fresh dict and
                # fail later with a type error naming the wrong thing.
                prefix = ".".join(parts[: depth + 1])
                raise ValueError(
                    f"invalid override {dotted_key!r}: {prefix!r} is a value "
                    f"({node[part]!r}), not a config section"
                )
            node = node[part]
        node[parts[-1]] = value
    return result


def deepcopy_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a plain nested dict/list structure."""
    if isinstance(data, dict):
        return {k: deepcopy_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [deepcopy_dict(v) for v in data]
    return data
