"""
Acoustic feature extraction from multichannel PCM frames.

All functions here are pure (no side effects, no state). See
docs/03_module_io.md section 2.1 and docs/05_algorithm.md section 1.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# A frequency band is a (low_hz, high_hz) tuple.
Band = Tuple[float, float]

# Default sub-band layout used for onset detection. Generic acoustic split,
# NOT tied to any specific game.
DEFAULT_BANDS: List[Band] = [
    (0.0, 200.0),       # low:  bass, body thumps
    (200.0, 2000.0),    # mid:  most footsteps energy
    (2000.0, 8000.0),   # high: transients, gunshots
    (8000.0, 24000.0),  # air:  crisp clicks, very high frequencies
]


def compute_channel_energy(samples: np.ndarray) -> np.ndarray:
    """Per-channel RMS^2 energy.

    Input:
      samples : shape=(frame_size, n_channels), float, range ~[-1, 1]
    Output:
      energy  : shape=(n_channels,), float64, range [0, +inf)
    """
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D (frame_size, n_channels), got shape {samples.shape}"
        )
    # mean(x^2) per column == RMS^2. Use float64 to avoid precision loss
    # on large accumulations.
    x = samples.astype(np.float64, copy=False)
    return np.mean(x * x, axis=0)


def compute_band_energy(
    samples: np.ndarray,
    sample_rate: int,
    bands: List[Band] = None,
) -> np.ndarray:
    """Sub-band energy via FFT, summed across all channels.

    Input:
      samples     : shape=(frame_size, n_channels), float, range ~[-1, 1]
      sample_rate : Hz, e.g. 48000
      bands       : list of (low_hz, high_hz); None -> DEFAULT_BANDS
    Output:
      band_energy : shape=(n_bands,), float64, range [0, +inf)

    Definition: for each band, sum the magnitude^2 spectrum over the
    frequencies inside [low_hz, high_hz], summed across channels. This is
    intentionally simple and game-agnostic.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D (frame_size, n_channels), got shape {samples.shape}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if bands is None:
        bands = DEFAULT_BANDS

    frame_size, n_channels = samples.shape
    if frame_size == 0:
        return np.zeros(len(bands), dtype=np.float64)

    # Mix down to mono for band-energy (we only use this for onset detection,
    # where per-channel detail is not needed). Mono = mean across channels.
    mono = samples.astype(np.float64, copy=False).mean(axis=1)

    # One-sided FFT magnitude. np.fft.rfft handles frame_size+1 // 2 bins.
    spectrum = np.fft.rfft(mono)
    mag_sq = (spectrum.real ** 2) + (spectrum.imag ** 2)

    # Frequency of each bin: k * fs / N
    n_bins = mag_sq.shape[0]
    freqs = np.arange(n_bins) * (sample_rate / frame_size)

    out = np.zeros(len(bands), dtype=np.float64)
    for i, (lo, hi) in enumerate(bands):
        if lo < 0 or hi <= lo:
            raise ValueError(f"invalid band {bands[i]!r}")
        mask = (freqs >= lo) & (freqs < hi)
        out[i] = mag_sq[mask].sum()
    return out


def compute_spectral_magnitude(
    samples: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mono magnitude spectrum (helper used by onset detector).

    Input:
      samples     : shape=(frame_size, n_channels)
      sample_rate : Hz
    Output:
      (freqs, mag): freqs shape=(n_bins,), mag shape=(n_bins,), float64, >= 0
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2-D")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    frame_size = samples.shape[0]
    if frame_size == 0:
        empty = np.zeros(1, dtype=np.float64)
        return empty, empty
    mono = samples.astype(np.float64, copy=False).mean(axis=1)
    spectrum = np.fft.rfft(mono)
    mag = np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2)
    freqs = np.arange(mag.shape[0]) * (sample_rate / frame_size)
    return freqs, mag