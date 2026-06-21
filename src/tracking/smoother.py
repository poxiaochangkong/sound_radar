"""
Contact smoother: turns instantaneous DirectionEstimates into decaying
RadarContact blips for the UI.

See docs/03_module_io.md section 3.1 and docs/05_algorithm.md section 5.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

from src.models import DirectionEstimate, RadarContact

# A new event that arrives within MERGE_ANGLE_DEG of an existing contact
# refreshes that contact instead of creating a new one.
_MERGE_ANGLE_DEG: float = 15.0

# Contacts below this intensity are pruned.
_PRUNE_INTENSITY: float = 0.02


def _circ_diff(a_deg: float, b_deg: float) -> float:
    """Smallest signed difference b - a in [-180, 180]."""
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


def _circ_lerp(a_deg: float, b_deg: float, alpha: float) -> float:
    """Circular linear interpolation from a toward b by alpha in [0,1]."""
    delta = _circ_diff(a_deg, b_deg)
    out = a_deg + alpha * delta
    # Wrap to [-180, 180).
    return (out + 180.0) % 360.0 - 180.0


class ContactSmoother:
    """Maintains a small set of decaying radar contacts.

    Each call to update():
      - ages existing contacts (exponential decay)
      - if `estimate` is provided, refreshes or creates a contact at its angle
      - prunes contacts whose intensity has decayed below a threshold
    Returns the list of currently visible contacts.
    """

    def __init__(
        self,
        decay_ms: int = 400,
        angle_smoothing: float = 0.35,
        clock: Optional[callable] = None,
    ) -> None:
        if decay_ms <= 0:
            raise ValueError(f"decay_ms must be > 0, got {decay_ms}")
        if not (0.0 <= angle_smoothing <= 1.0):
            raise ValueError(
                f"angle_smoothing must be in [0,1], got {angle_smoothing}"
            )
        self._half_life_s = decay_ms / 1000.0
        self._alpha = float(angle_smoothing)
        self._clock = clock or time.monotonic
        self._contacts: List[RadarContact] = []
        self._last_update: float = self._clock()

    def update(self, estimate: Optional[DirectionEstimate]) -> List[RadarContact]:
        """Advance the smoother by one frame.

        Input:
          estimate : current DirectionEstimate or None (no event this frame)
        Output:
          list of currently-active RadarContact (intensity > prune threshold)
        """
        now = self._clock()
        dt = max(0.0, now - self._last_update)
        self._last_update = now

        # ---- 1. Age existing contacts (exponential decay). ----
        if dt > 0 and self._half_life_s > 0:
            decay_factor = 0.5 ** (dt / self._half_life_s)
        else:
            decay_factor = 1.0

        surviving: List[RadarContact] = []
        for c in self._contacts:
            new_intensity = c.intensity * decay_factor
            if new_intensity >= _PRUNE_INTENSITY:
                surviving.append(RadarContact(
                    angle_deg=c.angle_deg,
                    intensity=new_intensity,
                    born_at=c.born_at,
                ))
        self._contacts = surviving

        # ---- 2. Apply the incoming estimate (refresh or create). ----
        if estimate is not None:
            self._merge_or_create(estimate, now)

        return list(self._contacts)

    def _merge_or_create(self, estimate: DirectionEstimate, now: float) -> None:
        # Find the closest existing contact within MERGE_ANGLE_DEG.
        best_idx = None
        best_dist = None
        for i, c in enumerate(self._contacts):
            d = abs(_circ_diff(c.angle_deg, estimate.angle_deg))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist <= _MERGE_ANGLE_DEG:
            c = self._contacts[best_idx]
            # Refresh: low-pass the angle toward the new estimate, reset
            # intensity and born_at.
            smoothed_angle = _circ_lerp(c.angle_deg, estimate.angle_deg,
                                        1.0 - self._alpha)
            self._contacts[best_idx] = RadarContact(
                angle_deg=smoothed_angle,
                intensity=1.0,
                born_at=now,
            )
        else:
            # New contact.
            self._contacts.append(RadarContact(
                angle_deg=estimate.angle_deg,
                intensity=1.0,
                born_at=now,
            ))

    def reset(self) -> None:
        self._contacts.clear()
        self._last_update = self._clock()

    @property
    def contacts(self) -> List[RadarContact]:
        return list(self._contacts)