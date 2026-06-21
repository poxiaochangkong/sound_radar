"""
Adaptive per-channel noise-floor estimation.

Algorithm: per-channel attack/release envelope follower.
  - When input < current estimate (release): slowly drift down.
  - When input >= current estimate (attack):  rise, but SLOWLY.

The attack time constant is intentionally LONG (default 2000 ms) so that
short transient events (gunshots ~200ms, footsteps ~100ms) do NOT drag the
noise floor up and kill their own SNR. This was a real bug at attack=50ms:
during a 10-frame gunshot burst, the noise floor rose to near the gunshot
energy within 2-3 frames, collapsing the SNR below threshold for the rest
of the burst.

Additionally, the caller can `freeze()` the estimator for a frame when an
onset is detected, so the transient frame is completely excluded from the
noise estimate.

See docs/03_module_io.md section 2.2 and docs/05_algorithm.md section 2.
"""
from __future__ import annotations

import numpy as np

# Numerical floor to keep log10 safe and avoid divide-by-zero downstream.
_FLOOR: float = 1e-10


class NoiseFloorEstimator:
    """Per-channel adaptive noise-floor estimator with freeze support."""

    def __init__(
        self,
        n_channels: int,
        attack_ms: int = 2000,
        release_ms: int = 5000,
        sample_rate: int = 48000,
        frame_size: int = 1024,
    ) -> None:
        if n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {n_channels}")
        if attack_ms <= 0:
            raise ValueError(f"attack_ms must be > 0, got {attack_ms}")
        if release_ms <= 0:
            raise ValueError(f"release_ms must be > 0, got {release_ms}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if frame_size <= 0:
            raise ValueError(f"frame_size must be > 0, got {frame_size}")

        self._n_channels = int(n_channels)
        self._estimate = np.full(self._n_channels, _FLOOR, dtype=np.float64)

        frame_seconds = frame_size / sample_rate
        # alpha = 1 - exp(-T / tau); smaller alpha = slower tracking.
        self._alpha_attack = float(1.0 - np.exp(-frame_seconds / (attack_ms / 1000.0)))
        self._alpha_release = float(1.0 - np.exp(-frame_seconds / (release_ms / 1000.0)))

        # When True, update() returns the current estimate without changing it.
        # Used to skip noise-floor updates on onset frames so transients don't
        # pollute the estimate.
        self._frozen = False

    @property
    def estimate(self) -> np.ndarray:
        """Current noise-floor estimate, shape=(n_channels,), >= 1e-10."""
        return self._estimate.copy()

    def freeze(self) -> None:
        """Mark the next update() as frozen (no change to the estimate)."""
        self._frozen = True

    def update(self, channel_energy: np.ndarray) -> np.ndarray:
        """Advance the estimator with one frame's per-channel RMS^2.

        Input:  channel_energy shape=(n_channels,), float, >= 0
        Output: noise_floor     shape=(n_channels,), float, >= 1e-10
        """
        if channel_energy.shape != (self._n_channels,):
            raise ValueError(
                f"channel_energy shape {channel_energy.shape} "
                f"!= ({self._n_channels},)"
            )
        if np.any(channel_energy < 0) or not np.all(np.isfinite(channel_energy)):
            raise ValueError("channel_energy must be non-negative and finite")

        # If frozen (e.g. onset frame), return current estimate unchanged.
        if self._frozen:
            self._frozen = False
            return self._estimate.copy()

        cur = self._estimate
        # Release branch: where input is quieter than current estimate.
        release_mask = channel_energy < cur
        # Attack branch: where input is at or above current estimate.
        attack_mask = ~release_mask

        # Release: drift down slowly toward the input.
        released = cur - self._alpha_release * (cur - channel_energy)
        # Attack: rise slowly toward the input (long time constant).
        attacked = cur + self._alpha_attack * (channel_energy - cur)

        new = np.where(release_mask, released, attacked)
        # Enforce the numerical floor and keep a copy for next frame.
        np.maximum(new, _FLOOR, out=new)
        self._estimate = new
        return new.copy()

    def reset(self) -> None:
        self._estimate.fill(_FLOOR)
        self._frozen = False