"""
VBAP-style multichannel direction estimation.

Core algorithm (pure function, no state, easily unit-testable):
  1. Skip ignored channels and channels with angle is None (e.g. LFE).
  2. Compute per-channel SNR vs noise floor.
  3. If no channel clears snr_threshold_db, return None (no trusted event).
  4. Take the channel with the highest SNR (A) and its best angular neighbor
     B among channels that ALSO clear the threshold.
  5. Weighted-average A and B's angles using SNR-linear weights.
  6. Confidence = how far max SNR exceeds the threshold, scaled by headroom.

See docs/03_module_io.md section 2.4 and docs/05_algorithm.md section 4.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.models import ChannelLayout, DirectionEstimate

# Small value added to energy/noise before division/log to avoid blow-ups.
_EPS: float = 1e-10
# SNR headroom (dB) above threshold at which confidence saturates to 1.0.
_DEFAULT_HEADROOM_DB: float = 18.0

# Channels ignored by default.
# LFE always carries no positional info (angle == None) so it is skipped
# regardless. We NO LONGER ignore C by default: many users want front-source
# accuracy, and ignoring C collapses front resolution to L/R (±15°).
# Callers can still pass ignore_channels=["C", "LFE"] explicitly if a game
# pushes lots of UI/ambient sound through C.
_DEFAULT_IGNORE: List[str] = ["LFE"]


def _angular_difference(a_deg: float, b_deg: float) -> float:
    """Signed smallest difference b - a in [-180, 180]."""
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


def _wrap_to_180(angle_deg: float) -> float:
    """Wrap an arbitrary angle (degrees) into [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def estimate_direction(
    channel_energy: np.ndarray,
    layout: ChannelLayout,
    noise_floor: np.ndarray,
    snr_threshold_db: float = 4.0,
    ignore_channels: Optional[List[str]] = None,
    headroom_db: float = _DEFAULT_HEADROOM_DB,
    timestamp: float = 0.0,
) -> Optional[DirectionEstimate]:
    """Estimate the bearing of a transient event from per-channel energy.

    Inputs:
      channel_energy : shape=(n_channels,), float, >= 0
      layout         : ChannelLayout whose .names aligns with channel_energy
      noise_floor    : shape=(n_channels,), float, > 0
      ignore_channels:
        None  -> use default (["LFE"]) — C is now INCLUDED for front accuracy.
        []    -> ignore nothing extra (LFE still excluded because angle is None)
        [...] -> ignore exactly these channel names (LFE always excluded)
    Output:
      DirectionEstimate or None.

    Guarantees when returning non-None:
      - angle_deg in [-180, 180]
      - confidence in [0, 1]
    """
    n = len(layout.names)
    if channel_energy.shape != (n,):
        raise ValueError(
            f"channel_energy shape {channel_energy.shape} != ({n},)"
        )
    if noise_floor.shape != (n,):
        raise ValueError(
            f"noise_floor shape {noise_floor.shape} != ({n},)"
        )
    if snr_threshold_db <= 0:
        raise ValueError(f"snr_threshold_db must be > 0, got {snr_threshold_db}")
    if headroom_db <= 0:
        raise ValueError(f"headroom_db must be > 0, got {headroom_db}")
    if np.any(channel_energy < 0):
        raise ValueError("channel_energy must be non-negative")
    if np.any(noise_floor <= 0):
        raise ValueError("noise_floor must be strictly positive")

    if ignore_channels is None:
        ignore_channels = _DEFAULT_IGNORE
    ignore_set = set(ignore_channels)

    # Build the list of candidate channels (real angle, not ignored).
    candidates: List[int] = []
    for i, name in enumerate(layout.names):
        angle = layout.angles.get(name)
        if angle is None:
            continue
        if name in ignore_set:
            continue
        candidates.append(i)

    if len(candidates) == 0:
        return None

    # Per-channel SNR (dB).
    snr_db = 10.0 * np.log10((channel_energy + _EPS) / (noise_floor + _EPS))

    # Channels that clear the threshold (the only ones that carry a trusted
    # bearing). Used both for the gate and for picking neighbor B.
    above = [i for i in candidates if snr_db[i] >= snr_threshold_db]
    if len(above) == 0:
        return None

    # ---- Dominant channel A: highest SNR among above-threshold channels. ----
    a_idx = max(above, key=lambda i: snr_db[i])
    a_angle = float(layout.angles[layout.names[a_idx]])
    a_snr = float(snr_db[a_idx])

    # If only one above-threshold channel, return it directly.
    if len(above) == 1:
        confidence = min(1.0, max(0.0, (a_snr - snr_threshold_db) / headroom_db))
        return DirectionEstimate(
            angle_deg=_wrap_to_180(a_angle),
            confidence=confidence,
            contributing_channels=[layout.names[a_idx]],
            timestamp=timestamp,
        )

    # ---- Neighbor B: nearest (circular) above-threshold channel to A. ----
    best_b_idx = None
    best_b_dist = None
    for ci in above:
        if ci == a_idx:
            continue
        c_angle = float(layout.angles[layout.names[ci]])
        dist = abs(_angular_difference(a_angle, c_angle))
        if best_b_dist is None or dist < best_b_dist:
            best_b_dist = dist
            best_b_idx = ci

    b_idx = best_b_idx
    b_angle = float(layout.angles[layout.names[b_idx]])
    b_snr = float(snr_db[b_idx])

    # ---- Weighted angular interpolation between A and B. ----
    w_a = max(0.0, a_snr - snr_threshold_db)
    w_b = max(0.0, b_snr - snr_threshold_db)
    if w_a + w_b < _EPS:
        # Degenerate (shouldn't happen since both are above threshold).
        return DirectionEstimate(
            angle_deg=_wrap_to_180(a_angle),
            confidence=0.0,
            contributing_channels=[layout.names[a_idx]],
            timestamp=timestamp,
        )

    delta = _angular_difference(a_angle, b_angle)
    fraction = w_b / (w_a + w_b)
    angle = a_angle + fraction * delta
    angle = _wrap_to_180(angle)

    confidence = min(1.0, max(0.0, (a_snr - snr_threshold_db) / headroom_db))

    if w_a >= w_b:
        contributing = [layout.names[a_idx], layout.names[b_idx]]
    else:
        contributing = [layout.names[b_idx], layout.names[a_idx]]

    return DirectionEstimate(
        angle_deg=angle,
        confidence=confidence,
        contributing_channels=contributing,
        timestamp=timestamp,
    )