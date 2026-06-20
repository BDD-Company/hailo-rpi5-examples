import pytest
from intercept_config import InterceptConfig


def test_defaults_load_from_yaml():
    cfg = InterceptConfig.load_defaults()
    assert cfg.distance_scale == 1.0
    assert cfg.guidance_lead is False
    assert cfg.lead_visual_dist == 12.0


def test_overrides_merge():
    cfg = InterceptConfig.from_overrides({"guidance_lead": True, "lead_speed": 18.0})
    assert cfg.guidance_lead is True
    assert cfg.lead_speed == 18.0
    assert cfg.distance_scale == 1.0   # untouched default


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        InterceptConfig.from_overrides({"not_a_real_key": 1})


def test_roundtrip_to_dict():
    cfg = InterceptConfig.load_defaults()
    d = cfg.to_dict()
    assert d["lead_visual_dist"] == 12.0
    assert InterceptConfig.from_overrides(d).to_dict() == d


def test_yaml_and_dataclass_keys_match():
    import yaml
    from dataclasses import fields
    from intercept_config import _YAML_PATH
    with open(_YAML_PATH) as f:
        ykeys = set(yaml.safe_load(f))
    fkeys = {fld.name for fld in fields(InterceptConfig)}
    assert ykeys == fkeys, f"missing in dataclass: {ykeys-fkeys}; missing in yaml: {fkeys-ykeys}"
