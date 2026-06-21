"""
Unit tests for feature extraction and noise-floor estimation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.features import compute_channel_energy, compute_band_energy
from src.analysis.noise_floor import NoiseFloorEstimator


def test_channel_energy_shape_and_sign():
    samples = np.random.default_rng(0).uniform(-1, 1, size=(1024, 8)).astype(np.float32)
    e = compute_channel_energy(samples)
    assert e.shape == (8,)
    assert np.all(e >= 0)


def test_channel_energy_silent_is_zero():
    samples = np.zeros((512, 4), dtype=np.float32)
    e = compute_channel_energy(samples)
    assert np.allclose(e, 0.0)


def test_channel_energy_full_scale_is_one():
    samples = np.ones((1024, 2), dtype=np.float32)
    e = compute_channel_energy(samples)
    assert np.allclose(e, 1.0)


def test_band_energy_nonneg_and_shape():
    samples = np.random.default_rng(1).uniform(-1, 1, size=(1024, 8)).astype(np.float32)
    b = compute_band_energy(samples, sample_rate=48000)
    assert b.shape == (4,)
    assert np.all(b >= 0)


def test_noise_floor_tracks_quiet_input():
    nf = NoiseFloorEstimator(n_channels=2, sample_rate=48000, frame_size=1024)
    # Feed quiet steady input for many frames; estimate should converge low.
    quiet = np.full(2, 1e-7, dtype=np.float64)
    for _ in range(200):
        nf.update(quiet)
    est = nf.estimate
    assert np.all(est > 0)
    assert np.all(est < 1e-5)


def test_noise_floor_resists_brief_transient():
    nf = NoiseFloorEstimator(n_channels=1, sample_rate=48000, frame_size=1024)
    quiet = np.array([1e-7], dtype=np.float64)
    # Warm up with quiet.
    for _ in range(200):
        nf.update(quiet)
    before = nf.estimate[0]
    # One loud transient frame.
    nf.update(np.array([1.0], dtype=np.float64))
    after_transient = nf.estimate[0]
    # Should rise only modestly (attack is fast but a single frame is small).
    assert after_transient > before
    assert after_transient < 0.5   # not dragged all the way to 1.0 in one frame
    # Back to quiet: should decay (slowly) but not increase.
    nf.update(quiet)
    assert nf.estimate[0] <= after_transient


def test_noise_floor_floor_enforced():
    nf = NoiseFloorEstimator(n_channels=2, sample_rate=48000, frame_size=1024)
    nf.update(np.zeros(2, dtype=np.float64))
    assert np.all(nf.estimate >= 1e-10)