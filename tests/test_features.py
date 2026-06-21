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
    # With the new slow attack (2000 ms), a single loud frame must NOT drag
    # the noise floor anywhere close to the loud value.
    nf = NoiseFloorEstimator(n_channels=1, sample_rate=48000, frame_size=1024)
    quiet = np.array([1e-7], dtype=np.float64)
    # Warm up with quiet.
    for _ in range(200):
        nf.update(quiet)
    before = nf.estimate[0]
    # One loud transient frame.
    nf.update(np.array([1.0], dtype=np.float64))
    after_transient = nf.estimate[0]
    # Should rise only very slightly (attack is now extremely slow).
    assert after_transient > before
    # Single frame must NOT move the estimate anywhere near 1.0.
    # With attack=2000ms and frame~21ms, alpha ~= 0.01, so one frame moves
    # the estimate by ~1% of (1.0 - before) ~ 0.01. Allow generous headroom.
    assert after_transient < 0.05, f"noise floor jumped to {after_transient}"
    # Back to quiet: should decay (slowly) but not increase.
    nf.update(quiet)
    assert nf.estimate[0] <= after_transient


def test_noise_floor_freeze_skips_update():
    # freeze() must make the next update() a no-op.
    nf = NoiseFloorEstimator(n_channels=1, sample_rate=48000, frame_size=1024)
    quiet = np.array([1e-7], dtype=np.float64)
    for _ in range(100):
        nf.update(quiet)
    before = nf.estimate[0]
    nf.freeze()
    # Feed a loud frame while frozen.
    out = nf.update(np.array([1.0], dtype=np.float64))
    # Estimate must be unchanged.
    assert abs(out[0] - before) < 1e-15
    assert abs(nf.estimate[0] - before) < 1e-15
    # The NEXT frame (not frozen) should update normally.
    nf.update(quiet)
    assert nf.estimate[0] != before


def test_noise_floor_floor_enforced():
    nf = NoiseFloorEstimator(n_channels=2, sample_rate=48000, frame_size=1024)
    nf.update(np.zeros(2, dtype=np.float64))
    assert np.all(nf.estimate >= 1e-10)