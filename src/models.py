"""
Shared data types for Sound Radar.

All dataclasses here are treated as IMMUTABLE snapshots passed across
threads. Producers must never mutate a dataclass after construction.

See docs/03_module_io.md for the full contract.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class AudioFrame:
    """One block of multichannel PCM captured from the loopback device.

    Invariants:
      - samples.ndim == 2
      - samples.shape[1] == len(channel_names)
      - samples.dtype == np.float32
      - np.all(np.isfinite(samples))
      - samples range roughly within [-1, 1] (may slightly overshoot)
    """
    samples: np.ndarray       # shape=(frame_size, n_channels), float32
    sample_rate: int          # e.g. 48000
    timestamp: float          # monotonic seconds (time.monotonic())
    channel_names: List[str]  # e.g. ["L","R","C","LFE","Ls","Rs","Lb","Rb"]


@dataclass
class FrameFeatures:
    """Per-frame acoustic features extracted from an AudioFrame.

    Invariants:
      - channel_energy.shape[0] == n_channels
      - band_energy.shape[0] == n_bands
      - all values >= 0 and finite
    """
    channel_energy: np.ndarray  # shape=(n_channels,), float64, RMS^2
    band_energy: np.ndarray     # shape=(n_bands,), float64
    timestamp: float


@dataclass
class DirectionEstimate:
    """Output of direction estimation for a single event.

    Invariants:
      - angle_deg in [-180, 180]
      - confidence in [0, 1]
    """
    angle_deg: float
    confidence: float
    contributing_channels: List[str]
    timestamp: float


@dataclass
class OnsetEvent:
    """A detected transient event from spectral-flux analysis."""
    timestamp: float
    strength: float    # spectral flux value, >= 0
    band_index: int    # which band the onset is dominant in


@dataclass
class RadarContact:
    """Display model consumed by the UI.

    Invariants:
      - angle_deg in [-180, 180]
      - intensity in [0, 1]
    """
    angle_deg: float
    intensity: float
    born_at: float


@dataclass
class ChannelLayout:
    """Speaker layout metadata: ordered channel names + per-channel angle.

    `angles` maps channel name -> angle in degrees (0 = front center,
    positive = clockwise) or None if the channel carries no positional
    information (e.g. LFE).
    """
    names: List[str]
    angles: dict   # {name: float | None}

    def angle_of(self, name: str) -> Optional[float]:
        return self.angles.get(name)


def channel_layout_for(layout_key: str, angles_map: dict) -> ChannelLayout:
    """Build a ChannelLayout from a config layout key (e.g. "7.1").

    Uses the canonical channel ordering matching the WASAPI / ITU convention
    for Windows multichannel PCM.
    """
    ordering = {
        "7.1":    ["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"],
        "5.1":    ["L", "R", "C", "LFE", "Ls", "Rs"],
        "stereo": ["L", "R"],
    }
    if layout_key not in ordering:
        raise ValueError(
            f"Unknown channel_layout '{layout_key}'. "
            f"Expected one of: {list(ordering.keys())}"
        )
    if layout_key not in angles_map:
        raise ValueError(
            f"No angle definition for channel_layout '{layout_key}' "
            f"in config.channel_angles"
        )
    names = ordering[layout_key]
    angles = angles_map[layout_key]
    missing = [n for n in names if n not in angles]
    if missing:
        raise ValueError(
            f"channel_angles['{layout_key}'] is missing entries "
            f"for channels: {missing}"
        )
    return ChannelLayout(names=names, angles=dict(angles))