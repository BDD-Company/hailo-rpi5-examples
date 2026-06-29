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

from config import Config, Range, Choices, MinItems, ExistingFile, _Constraint
from parse_config import parse_config, load_config, loads_config, ConfigError, check_schema
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
    threshold: Annotated[Optional[float], Range(min=0.0)]          # optional scalar (auto-None when omitted)

    # required (no default)
    gain:   Annotated[XY, Range(min=0.0)]                    # required XY
    data_file: ExistingFile                                  # required: coerces to Path, file must exist

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
    class OptionA:
        a : int = 0
        foo : str = ''
        quix : float = 0.0

    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class OptionB:
        b : int
        bar : str
        quix : float

    option : OptionA | OptionB = dataclasses.field(default_factory=lambda: TestConfig.OptionA())
    option_no_default : OptionA | OptionB
    optional_option : Optional[OptionA | OptionB]

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
OptionA = TestConfig.OptionA
OptionB = TestConfig.OptionB
Item = TestConfig.Item

# Path to THIS test file: a file guaranteed to exist while the tests run, used as
# a valid value for the required ExistingFile field (which coerces a config string
# to a Path and requires the file to exist, like Config.Inference.hef_model_path).
THIS_FILE = str(Path(__file__).resolve())


TEST_CONFIG_YAML = f"""\
name: hello
ratio: 0.5
count: 3
color: blue
point: [0.5, 0.5]
threshold: null
gain: [1.0, 2.0]
data_file: {THIS_FILE}
sub_section:
  required_value: 4.0
  label: b
  limits:
    lo: 0.1
    hi: 0.9
feature:
  intensity: 0.7
  mode: auto
option:                  # union A|B, OptionA-shaped -> resolves to OptionA
  a: 1
  foo: hello
  quix: 0.5
option_no_default:       # union A|B, OptionB-shaped -> falls through to OptionB
  b: 2
  bar: world
  quix: 1.5
optional_option: null    # Optional[A|B], absent feature -> None
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
__DEBUG_PRINT_ERROR_MESSAGES = False

def covert_to_config(data):
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
    cfg = covert_to_config(tvalid())
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
    assert isinstance(cfg.option, OptionA)          # union resolved to 1st variant
    assert cfg.option.a == 1
    assert isinstance(cfg.option_no_default, OptionB)   # resolved to 2nd variant
    assert cfg.option_no_default.b == 2
    assert cfg.optional_option is None              # Optional[union], null -> None
    assert [i.item_id for i in cfg.items] == [0, 1]
    assert cfg.items[0].angle == XY(90, 60)


def test_defaults_apply_when_omitted():
    d = tvalid()
    del d['name']          # has a default
    d['sub_section'].pop('label', None)   # nested default
    cfg = covert_to_config(d)
    assert cfg.name == "default"
    assert cfg.sub_section.label == 'a'


# ---------------------------------------------------------------------------
# Negative: type / bound / choices / enum / bulk / unknown-key.
# ---------------------------------------------------------------------------
def test_type_error_reported():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'ratio': "high"})
    assert any('ratio' in p and 'number' in p for p in ei.value.problems)


def test_bound_error_reported():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'ratio': 1.5, 'count': 0})
    probs = ei.value.problems
    assert any('ratio' in p and 'maximum' in p for p in probs)
    assert any('count' in p and 'minimum' in p for p in probs)


def test_choices_validation():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('sub_section', label='z'))
    assert any('sub_section.label' in p and 'not one of' in p for p in ei.value.problems)


def test_enum_validation():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'color': 'mauve'})
    assert any('color' in p and 'not one of' in p for p in ei.value.problems)


def test_errors_accumulated_in_bulk():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({
            'ratio': 2.0,            # bound
            'count': "x",            # type
            'color': 'mauve',        # enum
            'totally_unknown': 1,    # unknown key
        })
    assert len(ei.value.problems) >= 4


def test_unknown_top_level_key():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'naem': "typo"})
    assert any('naem' in p and 'unknown' in p for p in ei.value.problems)


def test_unknown_nested_key():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'sub_section': {'required_value': 1.0, 'bogus': 1}})
    assert any('sub_section.bogus' in p and 'unknown' in p for p in ei.value.problems)


def test_bool_is_not_int():
    with pytest.raises(ConfigError):
        covert_to_config(tvalid_with(count=True))


# ---------------------------------------------------------------------------
# XY handling, Optional scalars, MinItems.
# ---------------------------------------------------------------------------
def test_xy_validation():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(gain=[1, 2, 3]))
    assert any('gain' in p for p in ei.value.problems)
    # mapping form works
    cfg = covert_to_config(tvalid_with(point={'x': 0.4, 'y': 0.6}))
    assert cfg.point == XY(0.4, 0.6)


def test_xy_component_bounds():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'point': [0.5, 1.5]})
    assert any('point.y' in p for p in ei.value.problems)


def test_optional_scalar_accepts_null_and_value():
    assert covert_to_config(tvalid_with(threshold=None)).threshold is None
    assert covert_to_config(tvalid_with(threshold=0.5)).threshold == 0.5


def test_list_requires_min_items():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(items=[]))
    assert any('items' in p and 'at least' in p for p in ei.value.problems)


# ---------------------------------------------------------------------------
# ExistingFile: a path string is coerced to a pathlib.Path and the file must
# exist (mirrors Config.Inference.hef_model_path). THIS test file is the always-
# present target; a bogus path is rejected with a clear, path-prefixed problem.
# ---------------------------------------------------------------------------
def test_existing_file_loads_when_present():
    cfg = covert_to_config(tvalid())            # data_file -> THIS_FILE (exists)
    assert isinstance(cfg.data_file, Path)
    assert cfg.data_file == Path(THIS_FILE)
    assert cfg.data_file.is_file()


def test_existing_file_errors_when_missing():
    missing = str(Path(THIS_FILE).with_name("definitely_missing_zzz.yaml"))
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(data_file=missing))
    assert any(p.startswith('data_file:') and 'does not point to an existing file' in p
               for p in ei.value.problems), ei.value.problems


# ---------------------------------------------------------------------------
# Required fields, frozen, runtime fields, line numbers.
# ---------------------------------------------------------------------------
def test_missing_required_top_level_reported():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({})
    probs = ei.value.problems
    # Non-optional no-default fields are required; `feature`/`optional_option`
    # are Optional and silently default to None (see test below), so they are
    # NOT among the missing-required complaints even on an empty config.
    for name in ('gain', 'sub_section', 'option_no_default', 'items'):
        assert any(p.startswith(f"{name}:") and 'missing required' in p for p in probs), name
    assert not any(p.startswith('feature:') and 'missing required' in p for p in probs)
    assert not any(p.startswith('optional_option:') and 'missing required' in p for p in probs)


def test_missing_required_nested_field_reported():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(sub_section={'label': 'a'}))   # drops required_value
    assert any(p.startswith('sub_section.required_value:') and 'missing required' in p
               for p in ei.value.problems)


def test_config_is_frozen():
    cfg = covert_to_config(tvalid())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.ratio = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sub_section.required_value = 1.0
    cfg2 = dataclasses.replace(cfg, ratio=0.1)   # replace() yields an updated copy
    assert cfg2.ratio == 0.1
    assert cfg.ratio == 0.5


def test_runtime_fields_rejected_from_file():
    with pytest.raises(ConfigError) as ei:
        covert_to_config({'DEBUG': True})
    assert any('DEBUG' in p and 'unknown' in p for p in ei.value.problems)
    # but settable programmatically (frozen -> via replace)
    cfg = dataclasses.replace(covert_to_config(tvalid()), DEBUG=True)
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
    cfg = covert_to_config(tvalid_section('feature', enabled=False))
    assert cfg.feature is None


def test_enabled_true_returns_object():
    cfg = covert_to_config(tvalid_section('feature', enabled=True))
    assert isinstance(cfg.feature, FeatureSection)


def test_enabled_is_not_reflected_on_the_dataclass():
    cfg = covert_to_config(tvalid_section('feature', enabled=True))
    assert not hasattr(cfg.feature, 'enabled')


def test_enabled_omitted_defaults_to_present():
    d = tvalid()
    d['feature'].pop('enabled', None)
    assert covert_to_config(d).feature is not None


def test_enabled_must_be_bool():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('feature', enabled="yes"))
    assert any('feature.enabled' in p and 'boolean' in p for p in ei.value.problems)


def test_enabled_on_non_optional_section_is_error():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('sub_section', enabled=True))
    assert any(p.startswith('sub_section.enabled:') and 'only valid for optional' in p
               for p in ei.value.problems)


def test_disabled_section_is_still_validated_bounds():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('feature', enabled=False, intensity=5.0))
    assert any('feature.intensity' in p and 'maximum' in p for p in ei.value.problems)


def test_disabled_section_is_still_validated_unknown_key():
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('feature', enabled=False, __bogus__=1))
    assert any('feature.__bogus__' in p and 'unknown' in p for p in ei.value.problems)


def test_section_method_survives_parsing():
    cfg = covert_to_config(tvalid_section('feature', enabled=True))
    kwargs = cfg.feature.public_kwargs()
    assert 'mode' not in kwargs
    assert kwargs['intensity'] == 0.7


# ---------------------------------------------------------------------------
# Union-typed fields (A | B | ...): parse_config tries each variant left-to-
# right and uses the first that coerces cleanly, raising only if none match.
# ---------------------------------------------------------------------------
def test_union_resolves_to_first_matching_variant():
    cfg = covert_to_config(tvalid_with(option={'a': 5, 'foo': 'hi', 'quix': 1.5}))
    assert isinstance(cfg.option, OptionA)       # OptionA-shaped -> 1st variant
    assert cfg.option.a == 5


def test_union_falls_through_to_second_variant():
    # OptionA can't hold b/bar (unknown keys) -> resolution falls through to OptionB.
    cfg = covert_to_config(tvalid_with(option={'b': 7, 'bar': 'xx', 'quix': 2.0}))
    assert isinstance(cfg.option, OptionB)
    assert (cfg.option.b, cfg.option.bar, cfg.option.quix) == (7, 'xx', 2.0)


def test_union_first_match_wins_and_preserves_type():
    # int before float: an int value stays an int (the leftmost variant wins)
    # rather than being widened by the float variant.
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Scalar:
        value: int | float | str = 0

    assert parse_config(Scalar, {'value': 7}).value == 7            # 1st: int
    assert type(parse_config(Scalar, {'value': 7}).value) is int
    assert parse_config(Scalar, {'value': 1.5}).value == 1.5        # 2nd: float
    assert parse_config(Scalar, {'value': 'hi'}).value == 'hi'      # 3rd: str


def test_union_supports_more_than_two_variants_and_reports_all():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Scalar:
        value: int | float | str = 0

    with pytest.raises(ConfigError) as ei:
        parse_config(Scalar, {'value': [1, 2]})    # a list matches no variant
    msg = "\n".join(ei.value.problems)
    assert 'value' in msg and 'does not match any of' in msg
    assert 'int' in msg and 'float' in msg and 'str' in msg   # every variant named


def test_union_rejects_value_matching_no_variant():
    # {a: <str>}: invalid for OptionA (a is int) and OptionB (a unknown; b/bar/quix missing).
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(option={'a': 'not_an_int'}))
    assert any('option' in p and 'does not match any of' in p
               for p in ei.value.problems), ei.value.problems


def test_union_does_not_leak_losing_variant_errors():
    # OptionB-shaped value: OptionA loses internally (b/bar are unknown to it),
    # but those failures must not surface -- the config parses cleanly.
    cfg = covert_to_config(tvalid_with(option={'b': 9, 'bar': 'zz', 'quix': 3.0}))
    assert isinstance(cfg.option, OptionB)
    assert cfg.option.b == 9


def test_union_default_factory_applies_when_omitted():
    d = tvalid()
    del d['option']                              # has default_factory -> OptionA()
    cfg = covert_to_config(d)
    assert isinstance(cfg.option, OptionA)
    assert cfg.option.a == 0                      # OptionA's own defaults


def test_required_union_missing_is_reported():
    d = tvalid()
    del d['option_no_default']                   # required (no default)
    with pytest.raises(ConfigError) as ei:
        covert_to_config(d)
    assert any(p.startswith('option_no_default:') and 'missing required' in p
               for p in ei.value.problems), ei.value.problems


def test_optional_union_accepts_null_and_value():
    assert covert_to_config(tvalid_with(optional_option=None)).optional_option is None
    cfg = covert_to_config(tvalid_with(optional_option={'b': 4, 'bar': 'q', 'quix': 1.0}))
    assert isinstance(cfg.optional_option, OptionB)


def test_non_optional_union_rejects_null():
    # option_no_default is A|B (no None member) -> null matches no variant.
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(option_no_default=None))
    assert any('option_no_default' in p and 'does not match any of' in p
               for p in ei.value.problems), ei.value.problems


# ---------------------------------------------------------------------------
# Optional fields are never required: an omitted Optional[...] (with no explicit
# default) silently defaults to None instead of raising "missing required".
# Covers Optional[dataclass] (`feature`) and Optional[union] (`optional_option`).
# ---------------------------------------------------------------------------
def test_optional_field_defaults_to_none_when_omitted():
    d = tvalid()
    del d['feature']            # Optional[FeatureSection], no explicit default
    del d['optional_option']    # Optional[OptionA | OptionB], no explicit default
    cfg = covert_to_config(d)   # must NOT raise
    assert cfg.feature is None
    assert cfg.optional_option is None


def test_optional_field_omitted_does_not_report_missing():
    d = tvalid()
    del d['feature']
    del d['optional_option']
    # parses clean: build it and confirm no error path was taken at all.
    cfg = covert_to_config(d)
    assert isinstance(cfg, TestConfig)


def test_optional_section_still_validated_when_present():
    # Optional doesn't mean "skip validation": a present-but-bad section still errors.
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_section('feature', intensity=5.0))   # out of [0, 1]
    assert any('feature.intensity' in p and 'maximum' in p for p in ei.value.problems)


# ---------------------------------------------------------------------------
# Schema validation (opt-in `validate_schema=True`): static checks on the
# dataclass schema ITSELF, run BEFORE parsing (fail fast). The headline rule:
# an Optional[...] field must not declare a default (Optional auto-defaults to
# None). Plus: unsupported field types and invalid literal defaults. Every
# problem is aggregated into one ConfigError.
# ---------------------------------------------------------------------------
def test_clean_schema_passes_validation():
    # TestConfig is the canonical consistent schema -> no problems...
    assert check_schema(TestConfig) == []
    # ...and validate_schema=True parses valid data without raising.
    cfg = parse_config(TestConfig, tvalid(), validate_schema=True)
    assert isinstance(cfg, TestConfig)


def test_schema_validation_is_off_by_default():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        x: Optional[int] = None       # inconsistent, but not checked unless opted-in
    assert parse_config(Bad, {}).x is None          # parses fine, no schema check
    assert parse_config(Bad, {'x': 5}).x == 5


def test_schema_flags_optional_with_none_default():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        x: Optional[int] = None
    assert any('x' in p and 'Optional' in p and 'default' in p for p in check_schema(Bad))
    with pytest.raises(ConfigError) as ei:
        parse_config(Bad, {}, validate_schema=True)   # raised even though data is valid
    assert any('x' in p and 'Optional' in p for p in ei.value.problems)


def test_schema_flags_optional_with_value_default():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Inner:
        n: int = 0
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        a: Optional[Inner] = 0        # `0` is not a valid Optional[Inner], and a default at all
    assert any(p.startswith('a:') and 'Optional' in p for p in check_schema(Bad))


def test_schema_validation_runs_before_parse_fail_fast():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        x: Optional[int] = None       # schema inconsistency
        req: int                      # required; absent in the data below
    with pytest.raises(ConfigError) as ei:
        parse_config(Bad, {}, validate_schema=True)
    # fail fast: only the schema problem is raised; the data was never parsed,
    # so the missing-required `req` is NOT among the problems.
    assert any('Optional' in p for p in ei.value.problems)
    assert not any('req' in p and 'missing required' in p for p in ei.value.problems)


def test_schema_validation_aggregates_all_problems():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        x: Optional[int] = None
        y: Optional[str] = "hi"
        z: Optional[float] = 1.0
    probs = check_schema(Bad)
    assert len(probs) >= 3
    for name in ('x', 'y', 'z'):
        assert any(p.startswith(f"{name}:") for p in probs), name
    with pytest.raises(ConfigError) as ei:
        parse_config(Bad, {}, validate_schema=True)
    assert len(ei.value.problems) >= 3


def test_schema_flags_unsupported_field_type():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        x: complex = 0j               # parser has no coercion for complex
    assert any('x' in p and 'unsupported' in p for p in check_schema(Bad))


def test_schema_flags_invalid_literal_default():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        n: Annotated[int, Range(0, 10)] = 99    # default violates its own bound
        s: int = "not an int"                   # default has the wrong type
    probs = check_schema(Bad)
    assert any(p.startswith('n:') and 'invalid default' in p for p in probs)
    assert any(p.startswith('s:') and 'invalid default' in p for p in probs)


def test_schema_detects_nested_dataclass_inconsistency():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Inner:
        bad: Optional[int] = None
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Outer:
        inner: Inner
    assert any('inner.bad' in p and 'Optional' in p for p in check_schema(Outer))


def test_schema_detects_inconsistency_in_union_variant():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class V1:
        ok: int = 0
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class V2:
        bad: Optional[int] = None     # inside the 2nd variant
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Outer:
        choice: V1 | V2
    assert any('bad' in p and 'Optional' in p for p in check_schema(Outer))


def test_schema_detects_inconsistency_in_list_item():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class ItemBad:
        bad: Optional[int] = None
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Outer:
        items: list[ItemBad]
    assert any('bad' in p and 'Optional' in p for p in check_schema(Outer))


def test_schema_validation_ignores_runtime_fields():
    @dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
    class Bad:
        # runtime fields are never read from the file -> their schema is irrelevant
        x: Optional[int] = dataclasses.field(default=None, metadata={'runtime': True})
    assert check_schema(Bad) == []


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


def _admits_none(ann):
    """True if the annotation is Optional[...] (a union including None)."""
    base = get_args(ann)[0] if get_origin(ann) is Annotated else ann
    return get_origin(base) in (Union, types.UnionType) and type(None) in get_args(base)


def required_names(cls):
    # Mirrors parse_config: a field is required only if it has no default/factory
    # AND is not Optional (Optional[...] silently defaults to None when omitted).
    hints = get_type_hints(cls, include_extras=True)
    return [f.name for f in dataclasses.fields(cls)
            if not f.metadata.get('runtime')
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            and not _admits_none(hints[f.name])]


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
               str: 12345, XY: "not_an_xy", ExistingFile: 12345}   # ExistingFile wants a path string
REQUIRED_TOP = required_names(TestConfig)


def test_schema_has_leaf_fields():
    assert len(LEAF_FIELDS) >= 8


def test_schema_has_required_fields():
    # Non-optional no-default fields are required; Optional[...] ones are not.
    assert {'gain', 'sub_section', 'option_no_default', 'items'} <= set(REQUIRED_TOP)
    assert not ({'feature', 'optional_option'} & set(REQUIRED_TOP))   # Optional -> not required


def test_full_config_builds_full_object():
    cfg = covert_to_config(tvalid())
    assert isinstance(cfg.sub_section, SubSection)
    assert isinstance(cfg.feature, FeatureSection)
    assert all(isinstance(i, Item) for i in cfg.items)


@pytest.mark.parametrize("name", REQUIRED_TOP, ids=lambda n: n)
def test_missing_each_required_top_level_field_is_reported(name):
    data = tvalid()
    del data[name]
    with pytest.raises(ConfigError) as ei:
        covert_to_config(data)
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
        covert_to_config(data)
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
        covert_to_config(data)
    assert _problem_for(_path_str(segments), ei.value.problems), ei.value.problems


_CHOICE_FIELDS = [(s, [c for c in cs if isinstance(c, Choices)][0])
                  for s, b, cs in LEAF_FIELDS if any(isinstance(c, Choices) for c in cs)]


@pytest.mark.parametrize("segments,choices", _CHOICE_FIELDS,
                         ids=[_path_str(s) for s, _ in _CHOICE_FIELDS])
def test_every_choices_field_rejects_unknown(segments, choices):
    data = _build_nested(segments, "__not_a_valid_choice__")
    with pytest.raises(ConfigError) as ei:
        covert_to_config(data)
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
        covert_to_config(section_data)
    assert any('__bogus__' in p and 'unknown' in p for p in ei.value.problems)


def test_deeply_nested_bound_error_uses_full_path():
    d = tvalid()
    d['sub_section']['limits'] = {'hi': 5.0}   # 2 levels deep, out of [0, 1]
    with pytest.raises(ConfigError) as ei:
        covert_to_config(d)
    assert any(p.startswith('sub_section.limits.hi:') and 'maximum' in p
               for p in ei.value.problems)


def test_missing_required_deep_section_reported():
    # `limits` (required) dropped from sub_section -> reported with full path.
    with pytest.raises(ConfigError) as ei:
        covert_to_config(tvalid_with(sub_section={'required_value': 4.0}))   # replaces section, no `limits`
    assert any(p.startswith('sub_section.limits:') and 'missing required' in p
               for p in ei.value.problems)


# ===========================================================================
# Production Config integration (intentionally coupled to config.yaml): a tiny
# smoke test that the shipped config still loads, plus the load/parse forwarders.
# ===========================================================================
CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


def _real_config_model_present() -> bool:
    """Strict ExistingFile validation means the real config.yaml only loads where
    the HEF model exists (deployed on the Pi, absent on dev hosts/CI). Used to
    skip the load-the-real-file tests when the model is missing."""
    try:
        raw = yaml.safe_load(CONFIG_YAML.read_text()) or {}
        hef = (raw.get('inference') or {}).get('hef_model_path')
        return bool(hef) and Path(hef).is_file()
    except Exception:
        return False


_requires_model = pytest.mark.skipif(
    not _real_config_model_present(),
    reason="real config.yaml needs the HEF model present (deployed on the Pi, absent on host/CI)",
)


@_requires_model
def test_real_config_yaml_loads():
    cfg = Config.load(CONFIG_YAML)
    assert isinstance(cfg, Config)
    # structural sanity only (no tuning values, to avoid churn)
    assert isinstance(cfg.camera, Config.Camera)
    assert isinstance(cfg.drone, Config.Drone)


def test_config_parse_and_load_forwarders():
    # Config.parse forwards to parse_config with Config — no model file needed.
    with pytest.raises(ConfigError):
        Config.parse({'totally_unknown_key': 1})
    # Config.load forwards too, but the real config.yaml needs the HEF present.
    if not _real_config_model_present():
        pytest.skip("HEF model absent (host/CI)")
    assert isinstance(Config.load(CONFIG_YAML), Config)


def test_drone_api_enum_parses_from_yaml_literal():
    # The API enum drives the DroneMover backend switch in app.py. Its values must
    # be the YAML literals (the parser coerces enums by value, `API(value)`), so
    # 'mavsdk'/'betaflight' parse and anything else is rejected. Parses the Drone
    # subsection directly — no HEF model needed.
    base = {"connection_string": "x", "config": {}}
    for literal, member in (("mavsdk", Config.Drone.API.mavsdk),
                            ("betaflight", Config.Drone.API.betaflight)):
        drone = parse_config(Config.Drone, {**base, "api": literal})
        assert drone.api is member
    # omitted -> default
    assert parse_config(Config.Drone, base).api is Config.Drone.API.mavsdk
    # unknown backend rejected with a helpful message
    with pytest.raises(ConfigError) as ei:
        parse_config(Config.Drone, {**base, "api": "ardupilot"})
    assert any(p.startswith("api:") for p in ei.value.problems), ei.value.problems


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
