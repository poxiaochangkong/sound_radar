"""
Unit tests for config loading and the complete hardcoded fallback.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import load_config, ConfigError


def test_loads_without_config_file():
    # No config.yaml path -> must fall back to complete hard defaults.
    cfg = load_config(path=None)
    assert cfg.audio.channel_layout == "7.1"
    assert len(cfg.channel_layout_obj.names) == 8
    assert cfg.direction.snr_threshold_db > 0
    # radar colors must all be present (the previous bug).
    for key in ("background", "grid_color", "sweep_color",
                "contact_color", "front_zone_color",
                "side_zone_color", "back_zone_color"):
        assert key in cfg.ui.radar, f"missing default radar.{key}"


def test_loads_real_config_file():
    cfg = load_config("config.yaml")
    assert cfg.audio.channel_layout == "7.1"
    assert "L" in cfg.channel_layout_obj.angles


def test_profile_overrides_defaults():
    import tempfile, yaml as _yaml
    profile = {
        "recommended": {
            "direction": {"snr_threshold_db": 20.0},
            "onset": {"flux_multiplier": 5.0},
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(profile, f)
        path = f.name
    try:
        cfg = load_config(path=None, profile_path=path)
        assert abs(cfg.direction.snr_threshold_db - 20.0) < 1e-9
        assert abs(cfg.onset.flux_multiplier - 5.0) < 1e-9
    finally:
        os.unlink(path)


def test_invalid_layout_raises():
    import tempfile, yaml as _yaml
    bad = {"audio": {"channel_layout": "9.9"}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(bad, f)
        path = f.name
    try:
        try:
            load_config(path)
            assert False, "expected ConfigError"
        except ConfigError:
            pass
    finally:
        os.unlink(path)