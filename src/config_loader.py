"""
Configuration loader and validator.

See docs/03_module_io.md section 5 for the contract.

Design note: `_HARD_DEFAULTS` is a COMPLETE fallback. The application can run
with NO config.yaml at all — the defaults cover every field required by
build_app_config, including channel_angles and UI colors.
"""
from dataclasses import dataclass
from typing import Optional

import yaml

from src.models import channel_layout_for, ChannelLayout


class ConfigError(Exception):
    """Raised when config.yaml or a profile has invalid values."""


# ---- Typed config sub-sections ----------------------------------------------

@dataclass
class AudioConfig:
    device: Optional[str]
    sample_rate: int
    frame_size: int
    channel_layout: str

@dataclass
class DirectionConfig:
    snr_threshold_db: float
    ignore_center_channel: bool

@dataclass
class OnsetConfig:
    flux_multiplier: float
    refractory_ms: int

@dataclass
class TrackingConfig:
    decay_ms: int
    angle_smoothing: float

@dataclass
class UIConfig:
    window_title: str
    width: int
    height: int
    always_on_top: bool
    frameless: bool
    radar: dict   # colors as-is from yaml

@dataclass
class AppConfig:
    audio: AudioConfig
    channel_angles: dict
    channel_layout_obj: ChannelLayout
    direction: DirectionConfig
    onset: OnsetConfig
    tracking: TrackingConfig
    ui: UIConfig


# ---- Hardcoded defaults (lowest priority; COMPLETE fallback) ----------------
# Every field required by _build_app_config is present here so the app runs
# even if config.yaml and any profile are both missing.

_HARD_DEFAULTS = {
    "audio": {
        "device": None,
        "sample_rate": 48000,
        "frame_size": 1024,
        "channel_layout": "7.1",
    },
    "channel_angles": {
        "7.1": {
            "L": -30, "R": 30, "C": 0, "LFE": None,
            "Ls": -110, "Rs": 110, "Lb": -150, "Rb": 150,
        },
        "5.1": {
            "L": -30, "R": 30, "C": 0, "LFE": None,
            "Ls": -110, "Rs": 110,
        },
        "stereo": {
            "L": -30, "R": 30,
        },
    },
    "direction": {
        "snr_threshold_db": 12.0,
        "ignore_center_channel": True,
    },
    "onset": {
        "flux_multiplier": 3.0,
        "refractory_ms": 120,
    },
    "tracking": {
        "decay_ms": 400,
        "angle_smoothing": 0.35,
    },
    "ui": {
        "window_title": "Sound Radar",
        "width": 480,
        "height": 480,
        "always_on_top": True,
        "frameless": False,
        "radar": {
            "background":       "#0a0e14",
            "grid_color":       "#1f2937",
            "sweep_color":      "#22d3ee",
            "contact_color":    "#f59e0b",
            "front_zone_color": "#10b981",
            "side_zone_color":  "#3b82f6",
            "back_zone_color":  "#ef4444",
        },
    },
}


# ---- Deep merge helpers -----------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base; overlay wins on conflict."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---- Validators -------------------------------------------------------------

_HEX_RE = __import__("re").compile(r"^#[0-9A-Fa-f]{6}$")

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)

def _check_positive(value, name: str) -> None:
    _check(isinstance(value, (int, float)) and value > 0, f"{name} must be > 0, got {value!r}")

def _check_int_positive(value, name: str) -> None:
    _check(isinstance(value, int) and value > 0, f"{name} must be a positive int, got {value!r}")

def _check_range(value, lo: float, hi: float, name: str) -> None:
    _check(isinstance(value, (int, float)) and lo <= value <= hi,
           f"{name} must be in [{lo}, {hi}], got {value!r}")

def _check_color(value, name: str) -> None:
    _check(isinstance(value, str) and _HEX_RE.match(value) is not None,
           f"{name} must be '#RRGGBB', got {value!r}")


# ---- Public API -------------------------------------------------------------

def load_config(path: Optional[str] = "config.yaml",
                profile_path: Optional[str] = None) -> AppConfig:
    """Load and validate configuration.

    Priority: profile > config.yaml (if it exists) > hardcoded defaults.
    If `path` is None or the file is missing, only the hardcoded defaults
    (plus an optional profile) are used — the app still runs.
    Raises ConfigError on any invalid value.
    """
    import os

    merged = _deep_merge({}, _HARD_DEFAULTS)

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, user_cfg)

    if profile_path and os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            prof = yaml.safe_load(f) or {}
        # Profile may wrap its overrides under "recommended".
        if "recommended" in prof:
            prof = prof["recommended"]
        merged = _deep_merge(merged, prof)

    return _build_app_config(merged)


def _build_app_config(cfg: dict) -> AppConfig:
    # ---- audio ----
    audio = cfg["audio"]
    _check_int_positive(audio.get("sample_rate"), "audio.sample_rate")
    _check_int_positive(audio.get("frame_size"), "audio.frame_size")
    layout = audio.get("channel_layout")
    _check(layout in ("7.1", "5.1", "stereo"),
           f"audio.channel_layout must be '7.1' | '5.1' | 'stereo', got {layout!r}")
    device = audio.get("device")
    _check(device is None or isinstance(device, str),
           f"audio.device must be null or string, got {device!r}")

    # ---- channel_angles ----
    channel_angles = cfg.get("channel_angles", {})
    _check(isinstance(channel_angles, dict), "channel_angles must be a mapping")
    layout_obj = channel_layout_for(layout, channel_angles)

    # ---- direction ----
    direction = cfg["direction"]
    _check_positive(direction.get("snr_threshold_db"), "direction.snr_threshold_db")
    _check(isinstance(direction.get("ignore_center_channel"), bool),
           "direction.ignore_center_channel must be bool")

    # ---- onset ----
    onset = cfg["onset"]
    _check_positive(onset.get("flux_multiplier"), "onset.flux_multiplier")
    _check_int_positive(onset.get("refractory_ms"), "onset.refractory_ms")

    # ---- tracking ----
    tracking = cfg["tracking"]
    _check_int_positive(tracking.get("decay_ms"), "tracking.decay_ms")
    _check_range(tracking.get("angle_smoothing"), 0.0, 1.0, "tracking.angle_smoothing")

    # ---- ui ----
    ui = cfg["ui"]
    _check(isinstance(ui.get("window_title"), str), "ui.window_title must be string")
    _check_int_positive(ui.get("width"), "ui.width")
    _check_int_positive(ui.get("height"), "ui.height")
    radar = ui.get("radar", {})
    for color_key in ("background", "grid_color", "sweep_color",
                      "contact_color", "front_zone_color",
                      "side_zone_color", "back_zone_color"):
        _check_color(radar.get(color_key), f"ui.radar.{color_key}")

    return AppConfig(
        audio=AudioConfig(
            device=device,
            sample_rate=int(audio["sample_rate"]),
            frame_size=int(audio["frame_size"]),
            channel_layout=layout,
        ),
        channel_angles=channel_angles,
        channel_layout_obj=layout_obj,
        direction=DirectionConfig(
            snr_threshold_db=float(direction["snr_threshold_db"]),
            ignore_center_channel=bool(direction["ignore_center_channel"]),
        ),
        onset=OnsetConfig(
            flux_multiplier=float(onset["flux_multiplier"]),
            refractory_ms=int(onset["refractory_ms"]),
        ),
        tracking=TrackingConfig(
            decay_ms=int(tracking["decay_ms"]),
            angle_smoothing=float(tracking["angle_smoothing"]),
        ),
        ui=UIConfig(
            window_title=str(ui["window_title"]),
            width=int(ui["width"]),
            height=int(ui["height"]),
            always_on_top=bool(ui["always_on_top"]),
            frameless=bool(ui["frameless"]),
            radar=radar,
        ),
    )