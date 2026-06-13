import json

from control_config_override import apply_overrides


def test_override_updates_existing_scalar_keys(tmp_path):
    base = {"pd_coeff_p": 3, "thrust_min": 0.7, "confidence_min": 0.4}
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"pd_coeff_p": 5.5, "thrust_min": 0.9}))
    out = apply_overrides(base, str(f))
    assert out["pd_coeff_p"] == 5.5
    assert out["thrust_min"] == 0.9
    assert out["confidence_min"] == 0.4   # untouched


def test_override_rejects_unknown_key(tmp_path):
    base = {"pd_coeff_p": 3}
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"nonexistent_param": 1}))
    try:
        apply_overrides(base, str(f))
        assert False, "expected KeyError"
    except KeyError as e:
        assert "nonexistent_param" in str(e)


def test_none_path_returns_base_unchanged():
    base = {"pd_coeff_p": 3}
    assert apply_overrides(base, None) is base
