"""Host-runnable tests for config parsing/validation (no hailo/mavsdk needed)."""

import dataclasses
import types
from pathlib import Path
from typing import Annotated, Union, get_args, get_origin, get_type_hints

import pytest

from config import (
    Config, ByteTrackSection, Camera, CameraEntry, Drone,
    Range, Choices, _Constraint,
)
from parse_config import parse_config, load_config, ConfigError
from helpers import XY


CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


def _valid_dict():
    """A complete, valid config mapping (the shipped one) to build cases from."""
    import yaml
    return yaml.safe_load(CONFIG_YAML.read_text())


def valid_with(**overrides):
    """Shipped config with top-level keys overridden/added."""
    d = _valid_dict()
    d.update(overrides)
    return d


def valid_section(section, **overrides):
    """Shipped config with one nested section's keys overridden/added."""
    d = _valid_dict()
    d[section] = {**d.get(section, {}), **overrides}
    return d


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
    assert isinstance(cfg.camera, Camera)
    assert isinstance(cfg.drone, Drone)
    assert isinstance(cfg.bytetrack, ByteTrackSection)
    assert cfg.camera.cameras[0].name == 'wide'
    assert cfg.camera.cameras[0].frame_angular_size_deg == XY(107, 85)
    assert cfg.drone.connection_string == 'usb'
    assert cfg.drone.config.upside_down_angle_deg == 130
    assert cfg.bytetrack.track_thresh == 0.3
    assert cfg.bytetrack.recovery_max_dist is None


def test_empty_config_reports_missing_required_sections():
    with pytest.raises(ConfigError) as ei:
        parse_config({})
    probs = ei.value.problems
    for name in ('camera', 'drone', 'bytetrack', 'pd_coeff_p'):
        assert any(p.startswith(f"{name}:") and 'missing required' in p for p in probs), name


def test_optional_scalars_still_default_when_omitted():
    # A complete config may still omit scalar knobs that carry sensible defaults.
    cfg = parse_config(_valid_dict())
    assert cfg.confidence_min == 0.4
    assert cfg.camera.switch_size_ema_alpha == 0.3  # not in the shipped file


def test_bytetrack_tracker_kwargs_excludes_target_lock():
    cfg = parse_config(_valid_dict())
    kwargs = cfg.bytetrack.tracker_kwargs()
    assert 'target_lock' not in kwargs
    assert kwargs['track_thresh'] == 0.3


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
    # but settable programmatically (the config is frozen, so via replace())
    cfg = dataclasses.replace(parse_config(_valid_dict()), DEBUG=True)
    assert cfg.DEBUG is True


def test_xy_validation():
    with pytest.raises(ConfigError) as ei:
        parse_config(valid_with(pd_coeff_p=[1, 2, 3]))
    assert any('pd_coeff_p' in p for p in ei.value.problems)
    # mapping form works
    cfg = parse_config(valid_with(aim_point={'x': 0.4, 'y': 0.6}))
    assert cfg.aim_point == XY(0.4, 0.6)


def test_xy_component_bounds():
    with pytest.raises(ConfigError) as ei:
        parse_config({'aim_point': [0.5, 1.5]})
    assert any('aim_point.y' in p for p in ei.value.problems)


def test_choices_validation():
    with pytest.raises(ConfigError) as ei:
        parse_config(valid_section('camera', video_format='JPEG'))
    assert any('video_format' in p for p in ei.value.problems)


def test_constraints_resolve_to_real_types_like_annotated():
    # Fields use Annotated[type, Constraint], so the base type is introspectable
    # via get_type_hints and the validator rides along as Annotated metadata.
    plain = get_type_hints(Config)
    assert plain['confidence_min'] is float
    assert plain['safe_takeoff_period_ns'] is int
    assert plain['estimation_3d_method'] is str
    assert plain['pd_coeff_p'] is XY
    assert get_type_hints(Camera)['cameras'] == list[CameraEntry]

    extras = get_type_hints(Config, include_extras=True)
    ann = extras['confidence_min']
    assert get_origin(ann) is Annotated
    assert get_args(ann)[0] is float
    meta = get_args(ann)[1]
    assert isinstance(meta, Range) and hasattr(meta, 'validate')


def test_range_validates_each_numeric_field_of_any_dataclass():
    # TODO #1: Range applies the bound to every numeric field of a dataclass,
    # not just XY. Verify with an ad-hoc 3-field dataclass.
    @dataclasses.dataclass
    class Triple:
        a: float = 0.0
        b: float = 0.0
        c: float = 0.0

    errors = []
    Range(0.0, 1.0).validate(Triple(0.5, 2.0, -1.0), "t", errors)
    assert any(e.startswith("t.b:") and "maximum" in e for e in errors)
    assert any(e.startswith("t.c:") and "minimum" in e for e in errors)
    assert not any(e.startswith("t.a:") for e in errors)


def test_camera_requires_at_least_one():
    with pytest.raises(ConfigError) as ei:
        parse_config(valid_section('camera', cameras=[]))
    assert any('cameras' in p and 'at least' in p for p in ei.value.problems)


def test_bool_is_not_int():
    with pytest.raises(ConfigError):
        parse_config(valid_with(safe_takeoff_period_ns=True))


def test_config_is_frozen():
    cfg = parse_config(_valid_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.confidence_min = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.drone.connection_string = 'usb'
    # replace() produces an updated copy without mutating the original
    cfg2 = dataclasses.replace(cfg, confidence_min=0.1)
    assert cfg2.confidence_min == 0.1
    assert cfg.confidence_min == 0.4


def test_optional_field_accepts_null_and_value():
    cfg = parse_config(valid_section('bytetrack', recovery_max_dist=None))
    assert cfg.bytetrack.recovery_max_dist is None
    cfg = parse_config(valid_section('bytetrack', recovery_max_dist=0.5))
    assert cfg.bytetrack.recovery_max_dist == 0.5


def test_missing_required_nested_section_reported():
    # drone present but its required `config` block omitted.
    with pytest.raises(ConfigError) as ei:
        parse_config(valid_with(drone={'connection_string': 'usb'}))
    assert any(p.startswith('drone.config:') and 'missing required' in p
               for p in ei.value.problems)


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
    """Split a field annotation into (base_type, [constraints]).

    The constraints DSL produces typing.Annotated; Optional is unwrapped.
    """
    if get_origin(ann) is Annotated:
        args = get_args(ann)
        base, consts = args[0], [m for m in args[1:] if isinstance(m, _Constraint)]
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
    hints = get_type_hints(cls, include_extras=True)
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
        hints = get_type_hints(cls, include_extras=True)
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


def required_names(cls):
    return [f.name for f in dataclasses.fields(cls)
            if not f.metadata.get('runtime')
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING]


# (cls, field_name) for every required top-level / nested-section field. Adding
# a new required field anywhere is automatically covered by the test below.
REQUIRED_TOP = [(Config, n) for n in required_names(Config)]


def test_schema_has_required_fields():
    # sanity: the complex sub-sections really are required (no silent defaults)
    names = {n for _, n in REQUIRED_TOP}
    assert {'camera', 'drone', 'bytetrack', 'pd_coeff_p'} <= names


@pytest.mark.parametrize("name", [n for _, n in REQUIRED_TOP], ids=lambda n: n)
def test_missing_each_required_top_level_field_is_reported(name):
    data = _valid_dict()
    del data[name]
    with pytest.raises(ConfigError) as ei:
        parse_config(data)
    assert any(p.startswith(f"{name}:") and 'missing required' in p
               for p in ei.value.problems), ei.value.problems


def test_full_config_builds_full_object():
    cfg = parse_config(_valid_dict())
    # every nested section is materialised
    assert isinstance(cfg.camera, Camera)
    assert isinstance(cfg.drone, Drone)
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
