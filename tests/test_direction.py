"""
Unit tests for VBAP direction estimation.

Covers the contract from docs/03_module_io.md section 2.4 and the
circular-angle edge cases that are easy to get wrong.
"""
import os
import sys

import numpy as np

# Allow running pytest from project root without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.direction import estimate_direction
from src.types import ChannelLayout


# Standard 7.1 layout, matching config.yaml
LAYOUT_71 = ChannelLayout(
    names=["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"],
    angles={"L": -30, "R": 30, "C": 0, "LFE": None,
            "Ls": -110, "Rs": 110, "Lb": -150, "Rb": 150},
)

# Baseline noise floor (same small value on every channel).
NF = np.full(8, 1e-6, dtype=np.float64)


def _energy(**active):
    """Build a per-channel energy vector with the given active channels.

    Non-specified channels get a baseline ambient value 1e-7 (below NF).
    Specified channels get value = NF * 10**(snr/10) with snr=20 dB by default.
    """
    out = np.full(8, 1e-7, dtype=np.float64)
    for name, val in active.items():
        idx = LAYOUT_71.names.index(name)
        if val == "loud":
            out[idx] = NF[idx] * 10 ** (20.0 / 10.0)   # 20 dB above NF
        else:
            out[idx] = val
    return out


def test_all_zero_returns_none():
    energy = np.zeros(8, dtype=np.float64)
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is None


def test_all_below_threshold_returns_none():
    # 5 dB SNR on every channel, below the 12 dB threshold.
    energy = NF * 10 ** (5.0 / 10.0)
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is None


def test_single_channel_L():
    energy = _energy(L="loud")
    # C is ignored by default, so L's nearest neighbor among the candidates
    # is R (delta 60 deg) since C is excluded.
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is not None
    assert result.contributing_channels[0] == "L"
    # Only L is above threshold, so the angle is exactly L's angle.
    assert abs(result.angle_deg - (-30.0)) < 1e-6


def test_two_equal_channels_L_and_R():
    # L and R equally loud: midpoint should be ~0 (front center).
    energy = _energy(L="loud", R="loud")
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is not None
    assert abs(result.angle_deg - 0.0) < 1e-6
    assert set(result.contributing_channels) == {"L", "R"}


def test_two_equal_channels_across_180():
    # Lb (-150) and Rb (+150): the true midpoint is 180 (== -180), NOT 0.
    # This is the crucial circular-edge case.
    energy = _energy(Lb="loud", Rb="loud")
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is not None
    # Midpoint should be at ±180.
    assert abs(abs(result.angle_deg) - 180.0) < 1e-6


def test_center_channel_ignored_by_default():
    # Only C is loud. C is ignored by default, so no candidate clears threshold.
    energy = _energy(C="loud")
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is None


def test_center_channel_used_when_not_ignored():
    # Caller explicitly disables all ignoring with ignore_channels=[].
    energy = _energy(C="loud")
    result = estimate_direction(
        energy, LAYOUT_71, NF, snr_threshold_db=12.0,
        ignore_channels=[],
    )
    assert result is not None
    assert result.contributing_channels[0] == "C"
    assert abs(result.angle_deg - 0.0) < 1e-6


def test_lfe_never_localizes():
    # Only LFE is loud — it has angle None, so it must never produce a result.
    energy = _energy(LFE="loud")
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is None


def test_confidence_in_range():
    energy = _energy(L="loud")
    result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_saturates():
    # Very loud L -> SNR way above threshold -> confidence saturates at 1.
    energy = _energy(L=NF[0] * 10 ** (40.0 / 10.0))
    result = estimate_direction(energy, LAYOUT_71, NF,
                                snr_threshold_db=12.0, headroom_db=18.0)
    assert result is not None
    assert abs(result.confidence - 1.0) < 1e-6


def test_angle_in_valid_range():
    # Stress many different active-channel combinations.
    rng = np.random.default_rng(42)
    for _ in range(50):
        energy = NF * 10 ** (rng.uniform(0, 25, size=8))
        result = estimate_direction(energy, LAYOUT_71, NF, snr_threshold_db=12.0)
        if result is not None:
            assert -180.0 <= result.angle_deg <= 180.0
            assert 0.0 <= result.confidence <= 1.0