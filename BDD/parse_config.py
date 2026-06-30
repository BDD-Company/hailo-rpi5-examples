#!/usr/bin/env python3
"""YAML -> Config parsing and validation.

Walks the Config dataclass schema defined in config.py: coerces and
bound-checks each value, accumulates *all* problems, and annotates them with
the offending line in the source file. See config.py for the schema itself.
"""

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin, get_type_hints
import types

from helpers import XY
from config import _Constraint, ExistingFile

T = TypeVar("T")


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


def _union_variants(hint):
    """For a Union / ``X | Y`` annotation, return ``(non_none_variants, allows_none)``.

    For a non-union annotation, return ``(None, False)``. The shapes:
      - ``Optional[X]`` / ``X | None`` -> ``([X], True)``  (single variant, nullable)
      - ``A | B``                      -> ``([A, B], False)`` (multi-variant)
      - ``A | B | None``               -> ``([A, B], True)``  (multi-variant, nullable)
    """
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        args = get_args(hint)
        return [a for a in args if a is not _NONE_TYPE], _NONE_TYPE in args
    return None, False


def _type_label(ann) -> str:
    """Human-readable name for a (possibly Annotated) variant type, for errors."""
    base, _ = _split_constraint(ann)
    return base.__name__ if isinstance(base, type) else str(base)


def _allows_none(ann) -> bool:
    """True if a (possibly Annotated) annotation admits None, i.e. Optional[...]."""
    base, _ = _split_constraint(ann)
    return _union_variants(base)[1]


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

    # Union handling. Two shapes flow through here: the nullable single type
    # (Optional[X] / X | None) and the genuine multi-type union (A | B | ...),
    # which is resolved by trying each variant left-to-right (_coerce_union).
    optional = False
    variants, allows_none = _union_variants(base)
    if variants is not None:
        if allows_none and value is None:
            return None
        if len(variants) > 1:
            return _coerce_union(value, variants, constraints, path, errors)
        # Single variant => plain Optional[X]: unwrap and coerce as X. `optional`
        # records that None was allowed (it drives the `enabled` section toggle).
        optional = allows_none
        base, inner_constraints = _split_constraint(variants[0])
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
    elif isinstance(base, type) and issubclass(base, Enum):
        # TODO: for enums that doesn't have string literals as constants, use constant names instead (case insensitive).
        # e.g.:
        # class MyEnum(Enum):
        #     FOO = 1
        #     BAR = 2.0
        # parse following string as values:
        # 'foo' as MyEnum.FOO, 'FOO' as MyEnum.FOO, 'Bar' as MyEnum.BAR, etc.

        # YAML carries the enum's *value* (e.g. 'numpy'); look up the member.
        if isinstance(value, base):
            coerced = value
        else:
            try:
                coerced = base(value)
            except ValueError:
                errors.append(f"{path}: {value!r} is not one of {[m.value for m in base]}")
    elif base is ExistingFile or (isinstance(base, type) and issubclass(base, Path)):
        # Path-typed fields coerce from a path string. ExistingFile is a factory
        # that raises when the file is missing; surface that (and any plain Path
        # construction error) as a config problem instead of crashing the parse.
        if not isinstance(value, (str, Path)):
            errors.append(f"{path}: expected a path string, got {type(value).__name__}")
        else:
            try:
                coerced = base(value)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    elif base is XY:
        coerced = _coerce_xy(value, path, errors)
    elif is_dataclass(base):
        # `enabled` is a file-only toggle (NOT a dataclass field; never stored on
        # the result). On an Optional section it decides presence: enabled=False
        # still fully validates the section but yields None, so a feature can be
        # switched off without commenting it out. Consumers test
        # `section is not None`, never `section.enabled`. On a NON-optional
        # section, `enabled` is not allowed and is reported as an error.
        enabled = True
        if isinstance(value, dict) and 'enabled' in value:
            raw_enabled = value['enabled']
            value = {k: v for k, v in value.items() if k != 'enabled'}
            if not optional:
                errors.append(f"{path}.enabled: 'enabled' is only valid for optional sections")
            elif not isinstance(raw_enabled, bool):
                errors.append(f"{path}.enabled: expected a boolean, got {type(raw_enabled).__name__}")
            else:
                enabled = raw_enabled
        coerced = _parse_dataclass(base, value, path, errors)
        if optional and enabled is False and coerced is not _INVALID:
            coerced = None
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


def _coerce_union(value, variants, constraints, path, errors):
    """Resolve a multi-type union (``A | B | ...``) by trying each variant in
    declaration order and taking the first that coerces cleanly.

    Each attempt coerces into a *throwaway* error list, so a variant that fails
    leaves no partial errors behind (e.g. the loser's unknown-key complaints
    never reach the caller). The first variant that yields a value with an empty
    error list wins; the union's own (outer) constraints are then applied to it.
    Only if *every* variant fails is a single union error reported.
    """
    for variant in variants:
        trial: list[str] = []
        coerced = _coerce(value, variant, path, trial)
        if coerced is not _INVALID and not trial:
            for c in constraints:
                c.validate(coerced, path, errors)
            return coerced
    names = " | ".join(_type_label(v) for v in variants)
    errors.append(f"{path}: value does not match any of: {names}")
    return _INVALID


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
    # An Optional[...] field (its annotation admits None) is never *required* in
    # the file: when omitted it silently defaults to None (the placeholder loop
    # below supplies it). Only non-optional no-default fields are reported missing.
    config_required = {n for n in required if not _allows_none(anns[n])}
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
        elif f.name in config_required:
            errors.append(f"{prefix}{f.name}: missing required configuration key")

    for key in data:
        if key not in valid_names or (key in valid_names and _runtime_field(cls, key)):
            errors.append(f"{prefix}{key}: unknown configuration key")

    # Supply None for every no-default field still absent. For Optional fields
    # this is the real value (they default to None on the success path); for a
    # genuinely-missing required field it's just a placeholder so construction
    # never raises (an error was already recorded, so the object is discarded).
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


# ---------------------------------------------------------------------------
# Schema validation: static checks on the dataclass *schema itself*, with no
# config data involved. Opt-in via parse_config(..., validate_schema=True),
# where it runs BEFORE parsing so a malformed schema fails fast. Walks the whole
# schema (nested dataclasses, union variants, list items) and accumulates every
# problem, so they can be raised together as one aggregate.
# ---------------------------------------------------------------------------
def _is_supported_leaf(base) -> bool:
    """Whether `_coerce` handles `base` as a (non-container) leaf type."""
    if base in (bool, int, float, str, XY, ExistingFile, Any):
        return True
    return isinstance(base, type) and issubclass(base, (Enum, Path))


def _walk_schema_type(base, path, errors, seen):
    """Recurse a constraint-stripped type, reaching every nested dataclass
    (through union variants and list items) and flagging unsupported leaf types."""
    variants, _ = _union_variants(base)
    if variants is not None:
        for v in variants:
            _walk_schema_type(_split_constraint(v)[0], path, errors, seen)
        return
    if _is_supported_leaf(base):                 # scalar / enum / Path / XY / ExistingFile
        return
    if is_dataclass(base):
        _walk_schema(base, path, errors, seen)
        return
    if get_origin(base) in (list, types.GenericAlias) or base is list:
        args = get_args(base)
        item = _split_constraint(args[0])[0] if args else Any
        _walk_schema_type(item, f"{path}[]", errors, seen)
        return
    errors.append(f"{path}: unsupported field type {base!r}")


def _walk_schema(cls, path, errors, seen):
    """Accumulate schema inconsistencies for dataclass `cls` into `errors`.

    Per field (recursing into nested dataclasses / unions / lists):
      - an Optional[...] field must NOT declare a default or default_factory:
        Optional already yields None when its key is omitted, so a default is
        redundant or contradictory (e.g. `a: Optional[A] = 0`, `a: Optional[A] = None`).
      - the field type must be one the parser can coerce (no unsupported type).
      - a non-optional scalar field's literal default must itself be a valid
        value for the field's type and constraints.
    Runtime fields are skipped (the parser never reads them from the file).
    """
    if cls in seen:                              # reused / recursive types: check once
        return
    seen.add(cls)
    try:
        anns = get_type_hints(cls, include_extras=True)
    except Exception as exc:                     # unresolved annotation -> schema bug
        errors.append(f"{path or cls.__name__}: cannot resolve type hints: {exc}")
        return
    for f in fields(cls):
        if f.metadata.get('runtime'):
            continue
        fpath = f"{path}.{f.name}" if path else f.name
        ann = anns[f.name]
        base, _ = _split_constraint(ann)
        if _allows_none(ann):
            if f.default is not MISSING or f.default_factory is not MISSING:
                errors.append(f"{fpath}: Optional field must not declare a default "
                              f"(it defaults to None when the key is omitted)")
        elif f.default is not MISSING and _is_supported_leaf(base):
            # Validate a plain scalar/enum/XY/Path literal default against its
            # own declared type and bounds (factories are left alone).
            trial: list[str] = []
            _coerce(f.default, ann, fpath, trial)
            for e in trial:
                detail = e[len(fpath) + 2:] if e.startswith(f"{fpath}: ") else e
                errors.append(f"{fpath}: invalid default {f.default!r}: {detail}")
        _walk_schema_type(base, fpath, errors, seen)


def check_schema(config_type: type) -> list[str]:
    """Return every schema inconsistency in `config_type` (empty list if clean).

    A pure schema check (no config data); see `_walk_schema` for the rules.
    """
    errors: list[str] = []
    _walk_schema(config_type, "", errors, set())
    return errors


def parse_config(config_type: type[T], data: dict | None,
                 line_map: dict[str, int] | None = None,
                 source: str = "config file", *,
                 validate_schema: bool = False) -> T:
    """Validate a parsed-YAML mapping against `config_type`, or raise ConfigError.

    `config_type` is the root dataclass the mapping is parsed into (e.g. Config).
    If `line_map` (path -> file line) is supplied, every reported problem is
    annotated with the offending line in `source`.

    If `validate_schema` is True, the dataclass schema is checked for internal
    inconsistencies (see `check_schema`) *before* any parsing: any problem fails
    fast with one aggregate ConfigError, even when the config data is itself fine.
    """
    if validate_schema:
        schema_problems = check_schema(config_type)
        if schema_problems:
            raise ConfigError(schema_problems)
    errors: list[str] = []
    cfg = _parse_dataclass(config_type, data or {}, "", errors)
    if errors:
        raise ConfigError(_with_line_numbers(errors, line_map, source) if line_map else errors)
    return cfg


def loads_config(config_type: type[T], text: str, source: str = "<config>", *,
                 validate_schema: bool = False) -> T:
    """Validate a YAML *string* against `config_type` (raises ConfigError).

    Unlike `parse_config` (which takes an already-parsed mapping and therefore
    can't reference line numbers), this composes the YAML so every error is
    annotated with its line in `source`. `validate_schema` is forwarded.
    """
    import yaml
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([f"<root>: top level must be a mapping, got {type(data).__name__}"])
    line_map: dict[str, int] = {}
    _collect_node_lines(yaml.compose(text), "", line_map)
    return parse_config(config_type, data, line_map, source=source, validate_schema=validate_schema)


def load_config(config_type: type[T], path: str | Path, *,
                validate_schema: bool = False) -> T:
    """Read a YAML file and validate it against `config_type` (raises ConfigError).

    Errors are reported in bulk and annotated with the offending file line.
    `validate_schema` is forwarded.
    """
    return loads_config(config_type, Path(path).read_text(), source=Path(path).name,
                        validate_schema=validate_schema)
