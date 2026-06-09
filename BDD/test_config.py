"""Host-runnable tests for config parsing/validation (no hailo/mavsdk needed)."""

import dataclasses
import inspect
import types
from pathlib import Path
from typing import Union, get_args, get_origin

import pytest

from config import (
    Config, ByteTrackSection, CameraSection, DroneSection,
    Range, Choices, _Constraint,
    ConfigError, parse_config, load_config,
)
from helpers import XY


CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


def test_shipped_yaml_parses_and_matches_legacy_values():
    cfg = load_config(CONFIG_YAML)
    assert isinstance(cfg, Config)
    assert cfg.confidence_min == 0.4
    assert cfg.thrust_takeoff == 1.0
    assert cfg.estimation_3d_method == 'numpy'
    assert cfg.pd_coeff_p == XY(8, 2)
    assert cfg.target_size_m == XY(2, 2)
    assert cfg.safe_takeoff_period_ns == 1_000_000_000
    # nested sections
    assert isinstance(cfg.camera, CameraSection)
    assert isinstance(cfg.drone, DroneSection)
    assert isinstance(cfg.bytetrack, ByteTrackSection)
    assert cfg.camera.cameras[0].name == 'wide'
    assert cfg.camera.cameras[0].frame_angular_size_deg == XY(107, 85)
    assert cfg.drone.connection_string == 'usb'
    assert cfg.drone.config.upside_down_angle_deg == 130
    assert cfg.bytetrack.track_thresh == 0.3
    assert cfg.bytetrack.recovery_max_dist is None


def test_empty_config_uses_defaults():
    cfg = parse_config({})
    assert cfg.confidence_min == 0.4
    assert cfg.camera.cameras[0].name == 'wide'


def test_bytetrack_tracker_kwargs_excludes_target_lock():
    cfg = parse_config({})
    kwargs = cfg.bytetrack.tracker_kwargs()
    assert 'target_lock' not in kwargs
    assert kwargs['track_thresh'] == 0.5


def test_type_error_reported():
    with pytest.raises(ConfigError) as ei:
        parse_config({'confidence_min': "high"})
    assert any('confidence_min' in p and 'number' in p for p in ei.value.problems)


def test_bound_error_reported():
    with pytest.raises(ConfigError) as ei:
        parse_config({'confidence_min': 1.5, 'thrust_max': -0.2})
    probs = ei.value.problems
    assert any('confidence_min' in p and 'maximum' in p for p in probs)
    assert any('thrust_max' in p and 'minimum' in p for p in probs)


def test_errors_accumulated_in_bulk():
    with pytest.raises(ConfigError) as ei:
        parse_config({
            'confidence_min': 2.0,        # bound
            'thrust_max': "x",            # type
            'estimation_3d_method': 'nope',  # choices
            'totally_unknown': 1,         # unknown key
        })
    assert len(ei.value.problems) >= 4


def test_unknown_top_level_key():
    with pytest.raises(ConfigError) as ei:
        parse_config({'confidance_min': 0.4})  # typo
    assert any('confidance_min' in p and 'unknown' in p for p in ei.value.problems)


def test_unknown_nested_key():
    with pytest.raises(ConfigError) as ei:
        parse_config({'drone': {'config': {'upside_down_angel_deg': 130}}})
    assert any('upside_down_angel_deg' in p and 'unknown' in p for p in ei.value.problems)


def test_runtime_fields_rejected_from_file():
    with pytest.raises(ConfigError) as ei:
        parse_config({'DEBUG': True})
    assert any('DEBUG' in p and 'unknown' in p for p in ei.value.problems)
    # but settable programmatically
    cfg = parse_config({})
    cfg.DEBUG = True
    assert cfg.DEBUG is True


def test_xy_validation():
    with pytest.raises(ConfigError) as ei:
        parse_config({'pd_coeff_p': [1, 2, 3]})
    assert any('pd_coeff_p' in p for p in ei.value.problems)
    # mapping form works
    cfg = parse_config({'aim_point': {'x': 0.4, 'y': 0.6}})
    assert cfg.aim_point == XY(0.4, 0.6)


def test_xy_component_bounds():
    with pytest.raises(ConfigError) as ei:
        parse_config({'aim_point': [0.5, 1.5]})
    assert any('aim_point.y' in p for p in ei.value.problems)


def test_choices_validation():
    with pytest.raises(ConfigError) as ei:
        parse_config({'camera': {'video_format': 'JPEG'}})
    assert any('video_format' in p for p in ei.value.problems)


def test_camera_requires_at_least_one():
    with pytest.raises(ConfigError) as ei:
        parse_config({'camera': {'cameras': []}})
    assert any('cameras' in p and 'at least' in p for p in ei.value.problems)


def test_bool_is_not_int():
    with pytest.raises(ConfigError):
        parse_config({'safe_takeoff_period_ns': True})


def test_optional_field_accepts_null_and_value():
    cfg = parse_config({'bytetrack': {'recovery_max_dist': None}})
    assert cfg.bytetrack.recovery_max_dist is None
    cfg = parse_config({'bytetrack': {'recovery_max_dist': 0.5}})
    assert cfg.bytetrack.recovery_max_dist == 0.5


def test_errors_annotated_with_file_line_numbers(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        "confidence_min: 1.7\n"                       # line 1
        "aim_point: [0.5, 1.5]\n"                     # line 2 (XY component -> this line)
        "camera:\n"
        "  cameras:\n"
        "    - camera_id: 0\n"
        "      frame_angular_size_deg: [400, 85]\n"   # line 6
        "drone:\n"
        "  config:\n"
        "    bogus_key: 1\n"                          # line 9
    )
    with pytest.raises(ConfigError) as ei:
        load_config(bad)
    probs = ei.value.problems
    assert any('confidence_min' in p and 'broken.yaml line 1' in p for p in probs)
    assert any('aim_point.y' in p and 'broken.yaml line 2' in p for p in probs)
    assert any('frame_angular_size_deg.x' in p and 'broken.yaml line 6' in p for p in probs)
    assert any('bogus_key' in p and 'broken.yaml line 9' in p for p in probs)


# ---------------------------------------------------------------------------
# Introspection-driven tests. These walk the Config schema, so any field added
# later is automatically validated (correct type-checking, bound-checking and
# unknown-key handling) without touching this file.
# ---------------------------------------------------------------------------
def _unwrap(ann):
    """Split a raw field annotation into (base_type, [constraints]).

    A constraint instance carries its own base type; Optional is unwrapped.
    """
    if isinstance(ann, _Constraint):
        base, consts = ann.base, [ann]
    else:
        base, consts = ann, []
    if get_origin(base) in (Union, types.UnionType):
        non_none = [a for a in get_args(base) if a is not type(None)]
        if len(non_none) == 1:
            base = non_none[0]
    return base, consts


def _list_item_type(base):
    if get_origin(base) is list or base is list:
        args = get_args(base)
        return args[0] if args else None
    return None


def iter_leaf_fields(cls, prefix=()):
    """Yield (path_segments, base_type, constraints) for every scalar/XY leaf.

    path_segments is a tuple of ('key', name) / ('list',) steps so we can build
    a minimal nested dict that sets exactly one deep field.
    """
    hints = inspect.get_annotations(cls)
    for f in dataclasses.fields(cls):
        if f.metadata.get('runtime'):
            continue
        base, consts = _unwrap(hints[f.name])
        seg = prefix + (('key', f.name),)
        if dataclasses.is_dataclass(base) and base is not XY:
            yield from iter_leaf_fields(base, seg)
            continue
        item = _list_item_type(base)
        if item is not None and dataclasses.is_dataclass(item):
            yield from iter_leaf_fields(item, seg + (('list',),))
            continue
        yield seg, base, consts


def all_dataclasses():
    seen, stack = set(), [Config]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        hints = inspect.get_annotations(cls)
        for f in dataclasses.fields(cls):
            base, _ = _unwrap(hints[f.name])
            if dataclasses.is_dataclass(base) and base is not XY:
                stack.append(base)
            else:
                item = _list_item_type(base)
                if item is not None and dataclasses.is_dataclass(item):
                    stack.append(item)
    return seen


def _build_nested(segments, value):
    if not segments:
        return value
    head, rest = segments[0], segments[1:]
    if head[0] == 'key':
        return {head[1]: _build_nested(rest, value)}
    return [_build_nested(rest, value)]   # 'list'


def _path_str(segments):
    out = ""
    for kind in segments:
        if kind[0] == 'key':
            out += ("." if out else "") + kind[1]
        else:
            out += "[0]"
    return out


def _problem_for(path, problems):
    """A problem belongs to `path` if it starts with it (covers XY `.x`/`.y`)."""
    return any(p.startswith(path + ":") or p.startswith(path + ".") for p in problems)


LEAF_FIELDS = list(iter_leaf_fields(Config))
WRONG_TYPED = {bool: "not_a_bool", int: "not_an_int", float: "not_a_float",
               str: 12345, XY: "not_an_xy"}


def test_schema_has_leaf_fields():
    # sanity: the introspection actually found a representative set
    assert len(LEAF_FIELDS) > 30


def test_every_field_has_a_default():
    # Guarantees parse_config({}) keeps working as fields are added.
    for cls in all_dataclasses():
        for f in dataclasses.fields(cls):
            has_default = (f.default is not dataclasses.MISSING
                           or f.default_factory is not dataclasses.MISSING)
            assert has_default, f"{cls.__name__}.{f.name} needs a default"


def test_empty_config_builds_full_object():
    cfg = parse_config({})
    # every nested section is materialised
    assert isinstance(cfg.camera, CameraSection)
    assert isinstance(cfg.drone, DroneSection)
    assert isinstance(cfg.bytetrack, ByteTrackSection)
    assert cfg.drone.config.use_set_attitude is False


@pytest.mark.parametrize("segments,base", [(s, b) for s, b, _ in LEAF_FIELDS],
                         ids=[_path_str(s) for s, _, _ in LEAF_FIELDS])
def test_every_field_rejects_wrong_type(segments, base):
    if base not in WRONG_TYPED:
        pytest.skip(f"no wrong-type sample for {base}")
    data = _build_nested(segments, WRONG_TYPED[base])
    with pytest.raises(ConfigError) as ei:
        parse_config(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


_RANGED = [(s, b, [c for c in cs if isinstance(c, Range)][0])
           for s, b, cs in LEAF_FIELDS if any(isinstance(c, Range) for c in cs)]


@pytest.mark.parametrize("segments,base,rng", _RANGED,
                         ids=[_path_str(s) for s, _, _ in _RANGED])
def test_every_ranged_field_rejects_out_of_bounds(segments, base, rng):
    if rng.max is not None:
        bad = rng.max + 1
    elif rng.min is not None:
        bad = rng.min - 1
    else:
        pytest.skip("unbounded Range")
    value = [bad, bad] if base is XY else bad
    data = _build_nested(segments, value)
    with pytest.raises(ConfigError) as ei:
        parse_config(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


_CHOICE_FIELDS = [(s, [c for c in cs if isinstance(c, Choices)][0])
                  for s, b, cs in LEAF_FIELDS if any(isinstance(c, Choices) for c in cs)]


@pytest.mark.parametrize("segments,choices", _CHOICE_FIELDS,
                         ids=[_path_str(s) for s, _ in _CHOICE_FIELDS])
def test_every_choices_field_rejects_unknown(segments, choices):
    data = _build_nested(segments, "__not_a_valid_choice__")
    with pytest.raises(ConfigError) as ei:
        parse_config(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


@pytest.mark.parametrize("section_data", [
    {'__bogus__': 1},
    {'camera': {'__bogus__': 1}},
    {'camera': {'cameras': [{'__bogus__': 1}]}},
    {'drone': {'__bogus__': 1}},
    {'drone': {'config': {'__bogus__': 1}}},
    {'bytetrack': {'__bogus__': 1}},
])
def test_unknown_key_rejected_in_every_section(section_data):
    with pytest.raises(ConfigError) as ei:
        parse_config(section_data)
    assert any('__bogus__' in p and 'unknown' in p for p in ei.value.problems)
