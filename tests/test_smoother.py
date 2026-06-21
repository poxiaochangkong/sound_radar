"""
Unit tests for ContactSmoother.

Uses an injectable clock to avoid wall-clock nondeterminism.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.smoother import ContactSmoother
from src.types import DirectionEstimate


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self.t = t0
    def __call__(self) -> float:
        return self.t


def _est(angle_deg: float, t: float) -> DirectionEstimate:
    return DirectionEstimate(
        angle_deg=angle_deg, confidence=1.0,
        contributing_channels=["L"], timestamp=t,
    )


def test_no_event_returns_empty():
    clk = FakeClock()
    s = ContactSmoother(decay_ms=400, angle_smoothing=0.35, clock=clk)
    assert s.update(None) == []


def test_event_creates_contact():
    clk = FakeClock(0.0)
    s = ContactSmoother(decay_ms=400, angle_smoothing=0.0, clock=clk)
    contacts = s.update(_est(-30.0, 0.0))
    assert len(contacts) == 1
    assert abs(contacts[0].angle_deg - (-30.0)) < 1e-6
    assert abs(contacts[0].intensity - 1.0) < 1e-6


def test_decay_after_one_half_life():
    clk = FakeClock(0.0)
    s = ContactSmoother(decay_ms=400, angle_smoothing=0.0, clock=clk)
    s.update(_est(-30.0, 0.0))
    # Advance by exactly one half-life (400 ms).
    clk.t = 0.4
    contacts = s.update(None)
    assert len(contacts) == 1
    # Intensity should be ~0.5 after one half-life.
    assert abs(contacts[0].intensity - 0.5) < 1e-6


def test_contact_pruned_after_long_time():
    clk = FakeClock(0.0)
    s = ContactSmoother(decay_ms=100, angle_smoothing=0.0, clock=clk)
    s.update(_est(-30.0, 0.0))
    # Advance well past several half-lives so intensity drops below 0.02.
    clk.t = 5.0
    contacts = s.update(None)
    assert contacts == []


def test_same_direction_refreshes():
    clk = FakeClock(0.0)
    s = ContactSmoother(decay_ms=400, angle_smoothing=0.0, clock=clk)
    s.update(_est(-30.0, 0.0))
    # Half-life later the contact is at 0.5 intensity.
    clk.t = 0.4
    s.update(None)
    # A new event at the same bearing should reset intensity to 1.0, not
    # create a second contact.
    clk.t = 0.5
    contacts = s.update(_est(-30.0, 0.5))
    assert len(contacts) == 1
    assert abs(contacts[0].intensity - 1.0) < 1e-6


def test_different_direction_creates_new_contact():
    clk = FakeClock(0.0)
    s = ContactSmoother(decay_ms=1000, angle_smoothing=0.0, clock=clk)
    s.update(_est(-30.0, 0.0))            # front-left
    clk.t = 0.01
    contacts = s.update(_est(150.0, 0.01))  # back-right, far from -30
    assert len(contacts) == 2
    angles = sorted(c.angle_deg for c in contacts)
    assert abs(angles[0] - (-30.0)) < 1e-6
    assert abs(angles[1] - 150.0) < 1e-6


def test_angle_smoothing_lowpass():
    clk = FakeClock(0.0)
    # alpha=0.5: new angle contributes 50% toward existing contact.
    s = ContactSmoother(decay_ms=1000, angle_smoothing=0.5, clock=clk)
    s.update(_est(-30.0, 0.0))    # establish contact at -30
    # New event at -20 (10 deg away, within MERGE_ANGLE_DEG=15).
    # Smoothed should move halfway: lerp(a=-30, b=-20, alpha=0.5) = -25.
    clk.t = 0.001
    contacts = s.update(_est(-20.0, 0.001))
    assert len(contacts) == 1
    assert abs(contacts[0].angle_deg - (-25.0)) < 1e-6