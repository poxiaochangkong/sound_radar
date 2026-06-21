"""
Generic transient (onset) detection based on spectral flux.

Key property: the detection threshold is ADAPTIVE — it is a multiplier on
the rolling MEDIAN flux, never an absolute energy value. This is what makes
the detector game-agnostic: it works equally well in a quiet or loud mix.

See docs/03_module_io.md section 2.3 and docs/05_algorithm.md section 3.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from src.types import AudioFrame, OnsetEvent


class OnsetDetector:
    """Spectral-flux onset detector with adaptive thresholding."""

    def __init__(
        self,
        flux_multiplier: float = 3.0,
        refractory_ms: int = 120,
        median_window_frames: int = 60,  # ~1 second at 60 Hz frame rate
    ) -> None:
        if flux_multiplier <= 0:
            raise ValueError(f"flux_multiplier must be > 0, got {flux_multiplier}")
        if refractory_ms <= 0:
            raise ValueError(f"refractory_ms must be > 0, got {refractory_ms}")
        if median_window_frames < 3:
            raise ValueError(
                f"median_window_frames must be >= 3, got {median_window_frames}"
            )

        self._flux_multiplier = float(flux_multiplier)
        self._refractory_s = refractory_ms / 1000.0
        self._median_window = int(median_window_frames)

        self._prev_spectrum: Optional[np.ndarray] = None
        self._flux_history: Deque[float] = deque(maxlen=self._median_window)
        self._last_onset_time: float = -1e9
        # Seed the history with a tiny non-zero value so the first-frame
        # median is well defined.
        self._flux_history.append(1e-12)

    def process(self, frame: AudioFrame, band_energy: np.ndarray) -> Optional[OnsetEvent]:
        """Process one frame. Returns an OnsetEvent if a transient is detected.

        Input:
          frame       : AudioFrame (shape=(frame_size, n_channels))
          band_energy : shape=(n_bands,), used to identify the dominant band
        Output:
          OnsetEvent or None.
        """
        samples = frame.samples
        if samples.ndim != 2 or samples.shape[0] == 0:
            return None

        # Mono magnitude spectrum (mean across channels).
        mono = samples.astype(np.float64, copy=False).mean(axis=1)
        spectrum = np.abs(np.fft.rfft(mono))

        if self._prev_spectrum is None:
            # First frame: nothing to compare against.
            self._prev_spectrum = spectrum
            return None

        # Spectral flux = sum of POSITIVE differences of magnitude.
        # Only rising energy counts as an onset.
        diff = spectrum - self._prev_spectrum
        flux = float(np.sum(np.maximum(diff, 0.0)))
        self._prev_spectrum = spectrum

        # Adaptive threshold: multiplier on the rolling median flux.
        median_flux = float(np.median(np.asarray(self._flux_history, dtype=np.float64)))
        self._flux_history.append(flux)

        threshold = self._flux_multiplier * max(median_flux, 1e-12)
        if flux < threshold:
            return None

        # Refractory period: ignore onsets arriving too soon after the last one.
        now = frame.timestamp
        if (now - self._last_onset_time) < self._refractory_s:
            return None

        # Identify the dominant band at this instant.
        if band_energy.size == 0:
            band_index = 0
        else:
            band_index = int(np.argmax(band_energy))

        self._last_onset_time = now
        return OnsetEvent(
            timestamp=now,
            strength=flux,
            band_index=band_index,
        )

    def reset(self) -> None:
        self._prev_spectrum = None
        self._flux_history.clear()
        self._flux_history.append(1e-12)
        self._last_onset_time = -1e9