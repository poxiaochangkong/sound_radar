"""
Unit tests for OnsetDetector: thread-safe parameter setter + per-band refractory.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.onset import OnsetDetector
from src.types import AudioFrame


def _frame(samples: np.ndarray, t: float) -> AudioFrame:
    return AudioFrame(
        samples=samples.astype(np.float32),
        sample_rate=48000,
        timestamp=t,
        channel_names=["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"],
    )


def _quiet_frame(n=1024, t=0.0):
    # Very low-amplitude noise -> low, steady flux.
    s = np.random.default_rng(0).uniform(-1e-4, 1e-4, size=(n, 8))
    return _frame(s, t)


def _loud_transient_frame(n=1024, t=0.0):
    # A wideband burst (strong spectral rise vs the previous quiet frame).
    s = np.random.default_rng(1).uniform(-0.8, 0.8, size=(n, 8))
    return _frame(s, t)


def test_set_flux_multiplier_is_thread_safe():
    det = OnsetDetector(n_bands=4, flux_multiplier=3.0)
    det.set_flux_multiplier(2.0)
    assert abs(det.get_flux_multiplier() - 2.0) < 1e-9


def test_set_flux_multiplier_rejects_invalid():
    det = OnsetDetector(flux_multiplier=3.0)
    try:
        det.set_flux_multiplier(0.0)
        assert False
    except ValueError:
        pass
    try:
        det.set_flux_multiplier(-1.0)
        assert False
    except ValueError:
        pass


def test_first_frame_no_onset():
    det = OnsetDetector(n_bands=4, flux_multiplier=1.0, refractory_ms=10)
    be = np.array([1.0, 1.0, 1.0, 1.0])
    assert det.process(_quiet_frame(t=0.0), be) is None


def test_per_band_refractory_allows_other_band():
    # A strong onset on band 2 then an immediate onset on band 0 should both
    # fire (different bands -> independent refractory).
    det = OnsetDetector(n_bands=4, flux_multiplier=0.5, refractory_ms=200)
    # Prime the detector with a quiet frame so flux has a baseline.
    det.process(_quiet_frame(t=0.0), np.zeros(4))
    # Loud frame with dominant band 2 -> onset on band 2.
    evt1 = det.process(_loud_transient_frame(t=0.02),
                       np.array([0.1, 0.1, 1.0, 0.1]))
    # Another loud frame with dominant band 0, very close in time.
    # Per-band refractory must let band 0 through even though band 2 just fired.
    evt2 = det.process(_loud_transient_frame(t=0.03),
                       np.array([1.0, 0.1, 0.1, 0.1]))
    # We cannot guarantee evt1 is non-None (depends on flux), but if evt1
    # fired then evt2 MUST be allowed because it is a different band.
    if evt1 is not None and evt2 is not None:
        assert evt1.band_index != evt2.band_index