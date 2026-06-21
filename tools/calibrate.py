"""
Auto-calibration tool: records a few seconds of audio and derives a
per-game/environment profile.

Flow (see docs/06_calibration.md section 3):
  1. Quiet phase (5s): user keeps the game silent. We estimate the per-channel
     noise floor baseline.
  2. Active phase (10s): user plays normally. We capture activity-energy
     statistics and spectral-flux values.
  3. We derive recommended parameters and write profiles/<name>.yaml.

Usage:
    python tools/calibrate.py --game csgo [--seconds-quiet 5] [--seconds-active 10]

This tool uses the same WASAPI loopback path as the live app, so the Windows
default device must already be configured for 7.1 (or 5.1).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from typing import List

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.capture import AudioCapture, AudioDeviceError, make_drop_on_full_queue
from src.config_loader import load_config
from src.types import AudioFrame


def _collect_seconds(frame_queue, seconds: float, label: str) -> List[AudioFrame]:
    """Collect frames for `seconds` seconds. Returns the captured frames."""
    print(f"[calibrate] {label} — capturing {seconds:.0f}s...", flush=True)
    deadline = time.monotonic() + seconds
    frames: List[AudioFrame] = []
    while time.monotonic() < deadline:
        try:
            frames.append(frame_queue.get(timeout=0.1))
        except queue.Empty:
            continue
    print(f"[calibrate] {label} — got {len(frames)} frames")
    return frames


def _per_channel_mean_energy(frames: List[AudioFrame]) -> np.ndarray:
    if not frames:
        raise RuntimeError("No frames captured (check audio device config).")
    energies = np.array([np.mean(f.samples ** 2, axis=0) for f in frames])
    return energies.mean(axis=0)


def _spectral_flux_series(frames: List[AudioFrame]) -> np.ndarray:
    """Per-frame spectral flux summed across channels (mono)."""
    fluxes = []
    prev_mag = None
    for f in frames:
        mono = f.samples.astype(np.float64).mean(axis=1)
        mag = np.abs(np.fft.rfft(mono))
        if prev_mag is not None:
            fluxes.append(float(np.sum(np.maximum(mag - prev_mag, 0.0))))
        prev_mag = mag
    return np.array(fluxes, dtype=np.float64) if fluxes else np.zeros(1)


def main() -> int:
    p = argparse.ArgumentParser(description="Sound Radar calibration tool")
    p.add_argument("--game", required=True, help="profile name, e.g. csgo")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seconds-quiet", type=float, default=5.0)
    p.add_argument("--seconds-active", type=float, default=10.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    frame_queue, on_frame, dropped = make_drop_on_full_queue(maxsize=64)
    capture = AudioCapture(
        device=cfg.audio.device,
        sample_rate=cfg.audio.sample_rate,
        frame_size=cfg.audio.frame_size,
        channel_layout=cfg.audio.channel_layout,
        on_frame=on_frame,
    )

    print(f"[calibrate] opening audio device (layout={cfg.audio.channel_layout})...")
    try:
        capture.start()
    except AudioDeviceError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    channel_names = capture.channel_names  # public property

    try:
        print("[calibrate] >>> STEP 1: keep the game SILENT for "
              f"{args.seconds_quiet:.0f}s (no movement, no gunfire)")
        quiet_frames = _collect_seconds(frame_queue,
                                         args.seconds_quiet, "QUIET")

        print("[calibrate] >>> STEP 2: play NORMALLY for "
              f"{args.seconds_active:.0f}s (walk, shoot, explosions)")
        active_frames = _collect_seconds(frame_queue,
                                          args.seconds_active, "ACTIVE")
    finally:
        capture.stop()

    # ---- compute statistics ----
    quiet_energy = _per_channel_mean_energy(quiet_frames)
    active_energy = _per_channel_mean_energy(active_frames)

    noise_floor_baseline = {
        name: float(val) for name, val in zip(channel_names, quiet_energy)
    }

    eps = 1e-10
    ratio_db = 10 * np.log10((active_energy.max() + eps) /
                              (np.median(quiet_energy) + eps))
    recommended_snr = float(max(8.0, min(25.0, ratio_db * 0.5)))

    fluxes = _spectral_flux_series(active_frames)
    if fluxes.size > 0 and np.median(fluxes) > 0:
        flux_mult = (np.median(fluxes) + 1.5 * fluxes.std()) / np.median(fluxes)
        recommended_flux = float(max(1.5, min(5.0, flux_mult)))
    else:
        recommended_flux = 3.0

    recommended_decay = 400

    profile = {
        "game": args.game,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "noise_floor_baseline": noise_floor_baseline,
            "active_energy_max": float(active_energy.max()),
            "quiet_energy_median": float(np.median(quiet_energy)),
            "snr_ratio_db": float(ratio_db),
        },
        "recommended": {
            "direction": {"snr_threshold_db": round(recommended_snr, 1)},
            "onset": {"flux_multiplier": round(recommended_flux, 2)},
            "tracking": {"decay_ms": recommended_decay},
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "profiles")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.game}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)

    print(f"[calibrate] wrote profile: {out_path}")
    print(f"[calibrate] recommended snr_threshold_db = {recommended_snr:.1f}")
    print(f"[calibrate] recommended flux_multiplier  = {recommended_flux:.2f}")
    print(f"[calibrate] use it with:  python src/main.py --profile {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())