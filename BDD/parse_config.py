#!/usr/bin/env python3
"""YAML -> Config parsing and validation.

Walks the Config dataclass schema defined in config.py: coerces and
bound-checks each value, accumulates *all* problems, and annotates them with
the offending line in the source file. See config.py for the schema itself.
"""

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints
import types

from helpers import XY
from config import Config, _Constraint


class ConfigError(ValueError):
    """Raised with every accumulated type/bound/unknown-key problem at once."""
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"Invalid configuration ({len(self.problems)} problem(s)):\n  - {joined}")


# Sentinel signalling a value failed to coerce; the field keeps its default.
_INVALID = object()
_NONE_TYPE = type(None)


def _split_constraint(ann):
    """Return (base_type, [constraints]) for a field annotation.

    Range/Choices/MinItems produce typing.Annotated, so the base type is the
    first arg and the validators are the Annotated metadata. A plain type has
    no constraints.
    """
    if get_origin(ann) is Annotated:
        args = get_args(ann)
        return args[0], [m for m in args[1:] if isinstance(m, _Constraint)]
    return ann, []


def _is_optional(hint):
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return _NONE_TYPE in get_args(hint)
    return False


def _non_none_arg(hint):
    return next(a for a in get_args(hint) if a is not _NONE_TYPE)


def _coerce_xy(value, path, errors):
    if isinstance(value, XY):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            errors.append(f"{path}: expected [x, y] of two numbers, got {value!r}")
            return _INVALID
        return XY(float(value[0]), float(value[1]))
    if isinstance(value, dict):
        if set(value) != {'x', 'y'} or any(
                isinstance(value[k], bool) or not isinstance(value[k], (int, float)) for k in ('x', 'y')):
            errors.append(f"{path}: expected mapping with numeric x and y, got {value!r}")
            return _INVALID
        return XY(float(value['x']), float(value['y']))
    errors.append(f"{path}: expected an XY ([x, y] or {{x, y}}), got {type(value).__name__}")
    return _INVALID


def _coerce(value, ann, path, errors):
    base, constraints = _split_constraint(ann)

    # Optional[...] / X | None
    if _is_optional(base):
        if value is None:
            return None
        base, inner_constraints = _split_constraint(_non_none_arg(base))
        constraints = constraints + inner_constraints

    coerced = _INVALID
    if base is bool:
        if not isinstance(value, bool):
            errors.append(f"{path}: expected a boolean, got {type(value).__name__}")
        else:
            coerced = value
    elif base is int:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: expected an integer, got {type(value).__name__}")
        else:
            coerced = value
    elif base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: expected a number, got {type(value).__name__}")
        else:
            coerced = float(value)
    elif base is str:
        if not isinstance(value, str):
            errors.append(f"{path}: expected a string, got {type(value).__name__}")
        else:
            coerced = value
    elif base is XY:
        coerced = _coerce_xy(value, path, errors)
    elif is_dataclass(base):
        coerced = _parse_dataclass(base, value, path, errors)
    elif get_origin(base) in (list, types.GenericAlias) or base is list:
        item_type = get_args(base)[0] if get_args(base) else Any
        if not isinstance(value, list):
            errors.append(f"{path}: expected a list, got {type(value).__name__}")
        else:
            items = []
            for i, item in enumerate(value):
                r = _coerce(item, item_type, f"{path}[{i}]", errors)
                items.append(r if r is not _INVALID else None)
            coerced = items
    else:
        errors.append(f"{path}: unsupported field type {base!r}")

    if coerced is not _INVALID:
        for c in constraints:
            c.validate(coerced, path, errors)
    return coerced


def _required_names(cls) -> set:
    """Field names that have no default (neither default nor default_factory)."""
    return {f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
            and not f.metadata.get('runtime')}


def _blank(cls):
    """Construct an instance, passing None for every required field so it never
    raises. Only used on the error path, where the result is discarded."""
    return cls(**{name: None for name in _required_names(cls)})


def _parse_dataclass(cls, data, path, errors):
    prefix = f"{path}." if path else ""
    required = _required_names(cls)
    if not isinstance(data, dict):
        errors.append(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
        return _blank(cls)

    anns = get_type_hints(cls, include_extras=True)
    valid_names = set()
    kwargs = {}
    for f in fields(cls):
        valid_names.add(f.name)
        if f.metadata.get('runtime'):
            # Never read runtime fields from the file; leave the default.
            continue
        if f.name in data:
            r = _coerce(data[f.name], anns[f.name], f"{prefix}{f.name}", errors)
            if r is not _INVALID:
                kwargs[f.name] = r
        elif f.name in required:
            errors.append(f"{prefix}{f.name}: missing required configuration key")

    for key in data:
        if key not in valid_names or (key in valid_names and _runtime_field(cls, key)):
            errors.append(f"{prefix}{key}: unknown configuration key")

    # Placeholder for any required field still missing, so construction (used to
    # return *something*) never raises; on the error path the object is discarded.
    for name in required:
        kwargs.setdefault(name, None)
    try:
        return cls(**kwargs)
    except Exception as exc:  # pragma: no cover - kwargs are pre-validated
        errors.append(f"{path or '<root>'}: could not build {cls.__name__}: {exc}")
        return _blank(cls)


def _runtime_field(cls, name) -> bool:
    for f in fields(cls):
        if f.name == name:
            return bool(f.metadata.get('runtime'))
    return False


def _collect_node_lines(node, prefix: str, out: dict[str, int]) -> None:
    """Walk a yaml.compose() node tree, recording 1-based file lines per key path."""
    import yaml
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            path = f"{prefix}.{key_node.value}" if prefix else str(key_node.value)
            out[path] = key_node.start_mark.line + 1
            _collect_node_lines(value_node, path, out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            path = f"{prefix}[{i}]"
            out[path] = item.start_mark.line + 1
            _collect_node_lines(item, path, out)


def _line_for_path(path: str, line_map: dict[str, int]) -> int | None:
    """Resolve an error path to a file line, falling back to the nearest parent.

    Synthetic sub-paths that have no YAML node of their own (an XY component
    like `aim_point.y`, or a list element's missing key) resolve to the line of
    their closest enclosing key/element.
    """
    p = path
    while p:
        if p in line_map:
            return line_map[p]
        cut = max(p.rfind('.'), p.rfind('['))
        if cut <= 0:
            break
        p = p[:cut]
    return line_map.get(p)


def _with_line_numbers(errors: list[str], line_map: dict[str, int], source: str) -> list[str]:
    """Append `(<source> line N)` to each error using its leading path token."""
    enriched = []
    for err in errors:
        path = err.split(":", 1)[0].strip()
        line = _line_for_path(path, line_map) if path and path != "<root>" else None
        enriched.append(f"{err} ({source} line {line})" if line else err)
    return enriched


def parse_config(data: dict | None, line_map: dict[str, int] | None = None,
                 source: str = "config file") -> Config:
    """Validate a parsed-YAML mapping and return a Config, or raise ConfigError.

    If `line_map` (path -> file line) is supplied, every reported problem is
    annotated with the offending line in `source`.
    """
    errors: list[str] = []
    cfg = _parse_dataclass(Config, data or {}, "", errors)
    if errors:
        raise ConfigError(_with_line_numbers(errors, line_map, source) if line_map else errors)
    return cfg


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config (raises ConfigError).

    Errors are reported in bulk and annotated with the offending file line.
    """
    import yaml
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([f"<root>: top level must be a mapping, got {type(data).__name__}"])
    line_map: dict[str, int] = {}
    _collect_node_lines(yaml.compose(text), "", line_map)
    return parse_config(data, line_map, source=Path(path).name)
