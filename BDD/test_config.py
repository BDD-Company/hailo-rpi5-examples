"""Tests for config parsing/validation (host-runnable; no hailo/mavsdk needed).

Most tests run against a self-contained ``TestConfig`` schema + its own YAML
constant, NOT the production ``Config``/``config.yaml`` — so changing the real
config does not break the parser/feature tests. A small set of clearly-marked
tests stay on the real Config: a smoke test that ``config.yaml`` still loads,
and the consumer-contract tests (which scan the real controllers' source).

IMPORTANT: if a feature is added/removed/changed in Config or parse_config.py,
it must be mirrored in ``TestConfig`` and covered here (see MEMORY).
"""

import dataclasses
import enum
import types
from pathlib import Path
from typing import Annotated, Optional, Union, get_args, get_origin, get_type_hints

import yaml
import pytest

from config import Config, Range, Choices, MinItems, _Constraint
from parse_config import parse_config, load_config, loads_config, ConfigError
from helpers import XY


# ===========================================================================
# TestConfig: a synthetic schema exercising every parser feature, decoupled
# from the production Config. Field names are arbitrary / feature-specific.
# ===========================================================================
class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class TestConfig:
    __test__ = False  # don't let pytest try to collect this as a test class

    # scalars with defaults + constraints
    name:   str = "default"
    ratio:  Annotated[float, Range(0.0, 1.0)] = 0.5          # bounded float
    count:  Annotated[int, Range(min=1)] = 3                 # bounded int
    color:  Color = Color.GREEN                              # bare enum field
    point:  Annotated[XY, Range(0.0, 1.0)] = \
        dataclasses.field(default_factory=lambda: XY(0.5, 0.5))   # XY + bound
    threshold: Annotated[Optional[float], Range(min=0.0)] = None   # optional scalar

    # required (no default)
    gain:   Annotated[XY, Range(min=0.0)]                    # required XY

    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class SubSection:
        @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
        class Limits:                                        # nested-within-nested section
            lo: Annotated[float, Range(0.0, 1.0)] = 0.0
            hi: Annotated[float, Range(0.0, 1.0)] = 1.0
        required_value: Annotated[float, Range(0.0, 10.0)]   # nested required
        label:          Annotated[str, Choices('a', 'b', 'c')] = 'a'   # Choices + default
        limits:         Limits                               # required deeper (2-level) section
    sub_section: SubSection                                  # required non-optional section

    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class FeatureSection:
        intensity: Annotated[float, Range(0.0, 1.0)] = 0.7
        mode:      str = "auto"

        def public_kwargs(self) -> dict:
            # a method that survives parsing (mirrors Config.ByteTrack.tracker_kwargs)
            return {k: v for k, v in dataclasses.asdict(self).items() if k != 'mode'}
    feature: Optional[FeatureSection]                        # Optional[dataclass] -> enabled toggle

    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Item:
        item_id: Annotated[int, Range(min=0)] = 0
        angle:   Annotated[XY, Range(min=0.0, min_inclusive=False, max=360.0)]   # required
    items: Annotated[list[Item], MinItems(1)]               # list + MinItems, required

    # runtime-only: never read from the file (providing it is an unknown key)
    DEBUG: bool = dataclasses.field(default=False, metadata={'runtime': True})


SubSection = TestConfig.SubSection
Limits = TestConfig.SubSection.Limits
FeatureSection = TestConfig.FeatureSection
Item = TestConfig.Item


TEST_CONFIG_YAML = """\
name: hello
ratio: 0.5
count: 3
color: blue
point: [0.5, 0.5]
threshold: null
gain: [1.0, 2.0]
sub_section:
  required_value: 4.0
  label: b
  limits:
    lo: 0.1
    hi: 0.9
feature:
  intensity: 0.7
  mode: auto
items:
  - item_id: 0
    angle: [90, 60]
  - item_id: 1
    angle: [14, 8]
"""


def tvalid() -> dict:
    """A complete, valid TestConfig mapping to build cases from."""
    return yaml.safe_load(TEST_CONFIG_YAML)

# just a littly halper to eye-validate that error messages are looking human-friendly
__DEBUG_PRINT_ERROR_MESSAGES = True

def tparse(data):
    # Round-trip through YAML text (not parse_config(dict)) so the parser builds
    # a line map and every error is annotated with its line — exactly like a
    # real config file. yaml.dump is the canonical source those line #s refer to.
    text = yaml.dump(data, sort_keys=False, default_flow_style=False)
    try:
        return loads_config(TestConfig, text, source="<<dict-object>>")
    except ConfigError as e:
        if __DEBUG_PRINT_ERROR_MESSAGES:
            print("\n----- config -----\n" + text + "----- error -----\n" + str(e) + "\n")
        raise


def tvalid_with(**overrides):
    d = tvalid()
    d.update(overrides)
    return d


def tvalid_section(section, **overrides):
    d = tvalid()
    d[section] = {**d.get(section, {}), **overrides}
    return d


# ---------------------------------------------------------------------------
# Positive: a complete config parses, with nesting/enum/optional/list resolved.
# ---------------------------------------------------------------------------
def test_full_valid_config_parses():
    cfg = tparse(tvalid())
    assert isinstance(cfg, TestConfig)
    assert cfg.name == "hello"
    assert cfg.ratio == 0.5
    assert cfg.color is Color.BLUE                  # enum coerced from value
    assert cfg.point == XY(0.5, 0.5)
    assert cfg.gain == XY(1.0, 2.0)
    assert cfg.threshold is None
    assert isinstance(cfg.sub_section, SubSection)
    assert cfg.sub_section.required_value == 4.0
    assert isinstance(cfg.sub_section.limits, Limits)   # 2-level nesting
    assert cfg.sub_section.limits.lo == 0.1
    assert isinstance(cfg.feature, FeatureSection)  # optional section present
    assert [i.item_id for i in cfg.items] == [0, 1]
    assert cfg.items[0].angle == XY(90, 60)


def test_defaults_apply_when_omitted():
    d = tvalid()
    del d['name']          # has a default
    d['sub_section'].pop('label', None)   # nested default
    cfg = tparse(d)
    assert cfg.name == "default"
    assert cfg.sub_section.label == 'a'


# ---------------------------------------------------------------------------
# Negative: type / bound / choices / enum / bulk / unknown-key.
# ---------------------------------------------------------------------------
def test_type_error_reported():
    with pytest.raises(ConfigError) as ei:
        tparse({'ratio': "high"})
    assert any('ratio' in p and 'number' in p for p in ei.value.problems)


def test_bound_error_reported():
    with pytest.raises(ConfigError) as ei:
        tparse({'ratio': 1.5, 'count': 0})
    probs = ei.value.problems
    assert any('ratio' in p and 'maximum' in p for p in probs)
    assert any('count' in p and 'minimum' in p for p in probs)


def test_choices_validation():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_section('sub_section', label='z'))
    assert any('sub_section.label' in p and 'not one of' in p for p in ei.value.problems)


def test_enum_validation():
    with pytest.raises(ConfigError) as ei:
        tparse({'color': 'mauve'})
    assert any('color' in p and 'not one of' in p for p in ei.value.problems)


def test_errors_accumulated_in_bulk():
    with pytest.raises(ConfigError) as ei:
        tparse({
            'ratio': 2.0,            # bound
            'count': "x",            # type
            'color': 'mauve',        # enum
            'totally_unknown': 1,    # unknown key
        })
    assert len(ei.value.problems) >= 4


def test_unknown_top_level_key():
    with pytest.raises(ConfigError) as ei:
        tparse({'naem': "typo"})
    assert any('naem' in p and 'unknown' in p for p in ei.value.problems)


def test_unknown_nested_key():
    with pytest.raises(ConfigError) as ei:
        tparse({'sub_section': {'required_value': 1.0, 'bogus': 1}})
    assert any('sub_section.bogus' in p and 'unknown' in p for p in ei.value.problems)


def test_bool_is_not_int():
    with pytest.raises(ConfigError):
        tparse(tvalid_with(count=True))


# ---------------------------------------------------------------------------
# XY handling, Optional scalars, MinItems.
# ---------------------------------------------------------------------------
def test_xy_validation():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_with(gain=[1, 2, 3]))
    assert any('gain' in p for p in ei.value.problems)
    # mapping form works
    cfg = tparse(tvalid_with(point={'x': 0.4, 'y': 0.6}))
    assert cfg.point == XY(0.4, 0.6)


def test_xy_component_bounds():
    with pytest.raises(ConfigError) as ei:
        tparse({'point': [0.5, 1.5]})
    assert any('point.y' in p for p in ei.value.problems)


def test_optional_scalar_accepts_null_and_value():
    assert tparse(tvalid_with(threshold=None)).threshold is None
    assert tparse(tvalid_with(threshold=0.5)).threshold == 0.5


def test_list_requires_min_items():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_with(items=[]))
    assert any('items' in p and 'at least' in p for p in ei.value.problems)


# ---------------------------------------------------------------------------
# Required fields, frozen, runtime fields, line numbers.
# ---------------------------------------------------------------------------
def test_missing_required_top_level_reported():
    with pytest.raises(ConfigError) as ei:
        tparse({})
    probs = ei.value.problems
    for name in ('gain', 'sub_section', 'feature', 'items'):
        assert any(p.startswith(f"{name}:") and 'missing required' in p for p in probs), name


def test_missing_required_nested_field_reported():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_with(sub_section={'label': 'a'}))   # drops required_value
    assert any(p.startswith('sub_section.required_value:') and 'missing required' in p
               for p in ei.value.problems)


def test_config_is_frozen():
    cfg = tparse(tvalid())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.ratio = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sub_section.required_value = 1.0
    cfg2 = dataclasses.replace(cfg, ratio=0.1)   # replace() yields an updated copy
    assert cfg2.ratio == 0.1
    assert cfg.ratio == 0.5


def test_runtime_fields_rejected_from_file():
    with pytest.raises(ConfigError) as ei:
        tparse({'DEBUG': True})
    assert any('DEBUG' in p and 'unknown' in p for p in ei.value.problems)
    # but settable programmatically (frozen -> via replace)
    cfg = dataclasses.replace(tparse(tvalid()), DEBUG=True)
    assert cfg.DEBUG is True


def test_errors_annotated_with_file_line_numbers(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        "ratio: 1.7\n"                 # line 1 (bound)
        "point: [0.5, 1.5]\n"          # line 2 (XY component -> .y)
        "gain: [1, 2]\n"
        "sub_section:\n"
        "  required_value: 4.0\n"
        "  bogus_key: 1\n"             # line 6 (unknown)
        "feature:\n"
        "  intensity: 0.5\n"
        "items:\n"
        "  - item_id: 0\n"
        "    angle: [90, 60]\n"
    )
    with pytest.raises(ConfigError) as ei:
        load_config(TestConfig, bad)
    probs = ei.value.problems
    assert any('ratio' in p and 'broken.yaml line 1' in p for p in probs)
    assert any('point.y' in p and 'broken.yaml line 2' in p for p in probs)
    assert any('sub_section.bogus_key' in p and 'broken.yaml line 6' in p for p in probs)


# ---------------------------------------------------------------------------
# Annotated introspection: fields keep a real base type + validator metadata.
# ---------------------------------------------------------------------------
def test_constraints_resolve_to_real_types_like_annotated():
    plain = get_type_hints(TestConfig)
    assert plain['ratio'] is float
    assert plain['count'] is int
    assert plain['color'] is Color
    assert plain['point'] is XY
    assert get_type_hints(Item)['item_id'] is int

    extras = get_type_hints(TestConfig, include_extras=True)
    ann = extras['ratio']
    assert get_origin(ann) is Annotated
    assert get_args(ann)[0] is float
    meta = get_args(ann)[1]
    assert isinstance(meta, Range) and hasattr(meta, 'validate')


def test_range_validates_each_numeric_field_of_any_dataclass():
    # Range applies the bound to every numeric field of a dataclass, not just XY.
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


# ---------------------------------------------------------------------------
# Optional sub-section quick-disable: `enabled` is a FILE-ONLY toggle (not a
# dataclass field). On an Optional[<dataclass>] section it decides presence
# (enabled=False -> validate but yield None); on a non-Optional section it is
# an error. Consumers test `section is not None`, never `section.enabled`.
# ---------------------------------------------------------------------------
def test_enabled_false_returns_none():
    cfg = tparse(tvalid_section('feature', enabled=False))
    assert cfg.feature is None


def test_enabled_true_returns_object():
    cfg = tparse(tvalid_section('feature', enabled=True))
    assert isinstance(cfg.feature, FeatureSection)


def test_enabled_is_not_reflected_on_the_dataclass():
    cfg = tparse(tvalid_section('feature', enabled=True))
    assert not hasattr(cfg.feature, 'enabled')


def test_enabled_omitted_defaults_to_present():
    d = tvalid()
    d['feature'].pop('enabled', None)
    assert tparse(d).feature is not None


def test_enabled_must_be_bool():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_section('feature', enabled="yes"))
    assert any('feature.enabled' in p and 'boolean' in p for p in ei.value.problems)


def test_enabled_on_non_optional_section_is_error():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_section('sub_section', enabled=True))
    assert any(p.startswith('sub_section.enabled:') and 'only valid for optional' in p
               for p in ei.value.problems)


def test_disabled_section_is_still_validated_bounds():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_section('feature', enabled=False, intensity=5.0))
    assert any('feature.intensity' in p and 'maximum' in p for p in ei.value.problems)


def test_disabled_section_is_still_validated_unknown_key():
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_section('feature', enabled=False, __bogus__=1))
    assert any('feature.__bogus__' in p and 'unknown' in p for p in ei.value.problems)


def test_section_method_survives_parsing():
    cfg = tparse(tvalid_section('feature', enabled=True))
    kwargs = cfg.feature.public_kwargs()
    assert 'mode' not in kwargs
    assert kwargs['intensity'] == 0.7


# ---------------------------------------------------------------------------
# Introspection-driven tests: walk the TestConfig schema, so any field added
# to it is automatically covered for type/bound/choices/unknown-key/required.
# ---------------------------------------------------------------------------
def _unwrap(ann):
    """Split a field annotation into (base_type, [constraints]); Optional unwrapped."""
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
    """Yield (path_segments, base_type, constraints) for every scalar/XY leaf."""
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


def required_names(cls):
    return [f.name for f in dataclasses.fields(cls)
            if not f.metadata.get('runtime')
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING]


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
        out += ("." if (out and kind[0] == 'key') else "") + (kind[1] if kind[0] == 'key' else "[0]")
    return out


def _problem_for(path, problems):
    """A problem belongs to `path` if it starts with it (covers XY `.x`/`.y`)."""
    return any(p.startswith(path + ":") or p.startswith(path + ".") for p in problems)


LEAF_FIELDS = list(iter_leaf_fields(TestConfig))
WRONG_TYPED = {bool: "not_a_bool", int: "not_an_int", float: "not_a_float",
               str: 12345, XY: "not_an_xy"}
REQUIRED_TOP = required_names(TestConfig)


def test_schema_has_leaf_fields():
    assert len(LEAF_FIELDS) >= 8


def test_schema_has_required_fields():
    assert {'gain', 'sub_section', 'feature', 'items'} <= set(REQUIRED_TOP)


def test_full_config_builds_full_object():
    cfg = tparse(tvalid())
    assert isinstance(cfg.sub_section, SubSection)
    assert isinstance(cfg.feature, FeatureSection)
    assert all(isinstance(i, Item) for i in cfg.items)


@pytest.mark.parametrize("name", REQUIRED_TOP, ids=lambda n: n)
def test_missing_each_required_top_level_field_is_reported(name):
    data = tvalid()
    del data[name]
    with pytest.raises(ConfigError) as ei:
        tparse(data)
    assert any(p.startswith(f"{name}:") and 'missing required' in p
               for p in ei.value.problems), ei.value.problems


@pytest.mark.parametrize("segments,base", [(s, b) for s, b, _ in LEAF_FIELDS],
                         ids=[_path_str(s) for s, _, _ in LEAF_FIELDS])
def test_every_field_rejects_wrong_type(segments, base):
    if isinstance(base, type) and issubclass(base, enum.Enum):
        bad = "__not_a_member__"            # invalid for any reasonable enum
    elif base in WRONG_TYPED:
        bad = WRONG_TYPED[base]
    else:
        pytest.skip(f"no wrong-type sample for {base}")
    data = _build_nested(segments, bad)
    with pytest.raises(ConfigError) as ei:
        tparse(data)
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
        tparse(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


_CHOICE_FIELDS = [(s, [c for c in cs if isinstance(c, Choices)][0])
                  for s, b, cs in LEAF_FIELDS if any(isinstance(c, Choices) for c in cs)]


@pytest.mark.parametrize("segments,choices", _CHOICE_FIELDS,
                         ids=[_path_str(s) for s, _ in _CHOICE_FIELDS])
def test_every_choices_field_rejects_unknown(segments, choices):
    data = _build_nested(segments, "__not_a_valid_choice__")
    with pytest.raises(ConfigError) as ei:
        tparse(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


@pytest.mark.parametrize("section_data", [
    {'__bogus__': 1},
    {'sub_section': {'__bogus__': 1}},
    {'sub_section': {'limits': {'__bogus__': 1}}},   # 2 levels deep
    {'feature': {'__bogus__': 1}},
    {'items': [{'__bogus__': 1}]},
])
def test_unknown_key_rejected_in_every_section(section_data):
    with pytest.raises(ConfigError) as ei:
        tparse(section_data)
    assert any('__bogus__' in p and 'unknown' in p for p in ei.value.problems)


def test_deeply_nested_bound_error_uses_full_path():
    d = tvalid()
    d['sub_section']['limits'] = {'hi': 5.0}   # 2 levels deep, out of [0, 1]
    with pytest.raises(ConfigError) as ei:
        tparse(d)
    assert any(p.startswith('sub_section.limits.hi:') and 'maximum' in p
               for p in ei.value.problems)


def test_missing_required_deep_section_reported():
    # `limits` (required) dropped from sub_section -> reported with full path.
    with pytest.raises(ConfigError) as ei:
        tparse(tvalid_with(sub_section={'required_value': 4.0}))   # replaces section, no `limits`
    assert any(p.startswith('sub_section.limits:') and 'missing required' in p
               for p in ei.value.problems)


# ===========================================================================
# Production Config integration (intentionally coupled to config.yaml): a tiny
# smoke test that the shipped config still loads, plus the load/parse forwarders.
# ===========================================================================
CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


def test_real_config_yaml_loads():
    cfg = Config.load(CONFIG_YAML)
    assert isinstance(cfg, Config)
    # structural sanity only (no tuning values, to avoid churn)
    assert isinstance(cfg.camera, Config.Camera)
    assert isinstance(cfg.drone, Config.Drone)


def test_config_parse_and_load_forwarders():
    # Config.parse / Config.load forward to parse_config/load_config with Config.
    assert isinstance(Config.load(CONFIG_YAML), Config)
    with pytest.raises(ConfigError):
        Config.parse({'totally_unknown_key': 1})


# ===========================================================================
# Consumer contract (real controllers): a feature backed by an Optional section
# must be gated on `section is not None`, NEVER on `section.enabled`. The
# controllers import hailo/mavsdk and can't be imported on host, so we scan
# their source instead.
# ===========================================================================
_CONSUMER_FILES = ('app.py', 'drone_controller.py', 'platform_controller.py')


def _optional_dataclass_sections(cls):
    """Fields of `cls` typed Optional[<dataclass>] — the toggleable sections."""
    out = []
    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        ann = hints[f.name]
        if get_origin(ann) not in (Union, types.UnionType):
            continue
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1 and dataclasses.is_dataclass(inner[0]):
            out.append(f.name)
    return out


def _strip_comments(src: str) -> str:
    return "\n".join(line.split('#', 1)[0] for line in src.splitlines())


def test_config_has_optional_toggle_sections():
    assert {'optical_refinement', 'bytetrack'} <= set(_optional_dataclass_sections(Config))


def test_consumers_gate_on_presence_not_enabled():
    sections = _optional_dataclass_sections(Config)
    base = CONFIG_YAML.parent
    for fn in _CONSUMER_FILES:
        code = _strip_comments((base / fn).read_text())
        for sec in sections:
            assert f".{sec}.enabled" not in code, (
                f"{fn} reads .{sec}.enabled; gate on `{sec} is not None` instead")
        assert ".use_byte_track" not in code, f"{fn} still references .use_byte_track"


def test_app_gates_tracker_on_bytetrack_presence():
    app_code = _strip_comments((CONFIG_YAML.parent / 'app.py').read_text())
    assert "config.bytetrack is not None" in app_code
