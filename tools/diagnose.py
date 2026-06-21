"""
Diagnostic tool: record N seconds of loopback audio and print per-channel
analysis to pinpoint which layer is failing.

Usage:
    # Record 5 seconds while you make sounds in the game (shoot, walk), then
    # see the per-channel energy breakdown.
    python tools/diagnose.py --seconds 5

What it tells you:
  - If channels Ls/Rs/Lb/Rb are all ~0 while L/R have signal -> the game is
    NOT outputting real 7.1 (Windows up-mixed stereo to 7.1 with empty
    surround channels). Check game audio settings.
  - If energy is well-distributed but SNR is low -> onset/direction thresholds
    need tuning.
  - If you see clear transients in the energy curve -> audio is fine, the
    problem is downstream (onset/direction/UI).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.capture import AudioCapture, AudioDeviceError, make_drop_on_full_queue
from src.config_loader import load_config
from src.models import AudioFrame

_CHANNEL_NAMES_71 = ["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"]


def main() -> int:
    p = argparse.ArgumentParser(description="Sound Radar diagnostic")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    frame_queue, on_frame, dropped = make_drop_on_full_queue(maxsize=128)
    capture = AudioCapture(
        device=cfg.audio.device,
        sample_rate=cfg.audio.sample_rate,
        frame_size=cfg.audio.frame_size,
        channel_layout=cfg.audio.channel_layout,
        on_frame=on_frame,
    )

    print(f"[diag] opening loopback device (layout={cfg.audio.channel_layout})...")
    try:
        capture.start()
    except AudioDeviceError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"[diag] >>> RECORDING {args.seconds:.1f}s — make sounds NOW (shoot, walk, etc.)")
    deadline = time.monotonic() + args.seconds
    frames: List[AudioFrame] = []
    while time.monotonic() < deadline:
        try:
            frames.append(frame_queue.get(timeout=0.1))
        except queue.Empty:
            continue
    capture.stop()

    if not frames:
        print("[diag] NO frames captured — audio stream did not deliver data.")
        print("       Check: is any application actually playing audio?")
        return 1

    print(f"\n[diag] captured {len(frames)} frames ({len(frames)*cfg.audio.frame_size/cfg.audio.sample_rate:.2f}s)")

    # ---- Per-channel overall energy ----
    all_samples = np.concatenate([f.samples for f in frames], axis=0)
    n_ch = all_samples.shape[1]
    names = _CHANNEL_NAMES_71[:n_ch]
    per_ch_energy = (all_samples ** 2).mean(axis=0)
    per_ch_peak = np.abs(all_samples).max(axis=0)

    print("\n=== Per-channel statistics over the whole recording ===")
    print(f"{'ch':<5} {'RMS^2':>14} {'peak':>10}  bar")
    max_e = max(per_ch_energy.max(), 1e-12)
    for i, name in enumerate(names):
        e = per_ch_energy[i]
        pk = per_ch_peak[i]
        bar = "#" * int(40 * e / max_e)
        print(f"{name:<5} {e:14.6e} {pk:10.4f}  {bar}")

    # ---- Per-frame peak detection (which frames were loud?) ----
    frame_energies = np.array([(f.samples ** 2).mean() for f in frames])
    loud_threshold = frame_energies.mean() + 3 * frame_energies.std() + 1e-12
    loud_frames = np.where(frame_energies > loud_threshold)[0]
    print(f"\n=== Loud-frame detection ===")
    print(f"mean frame energy = {frame_energies.mean():.6e}")
    print(f"loud threshold    = {loud_threshold:.6e} (mean + 3*std)")
    print(f"loud frames count = {len(loud_frames)} / {len(frames)}")

    if len(loud_frames) > 0:
        print("\n=== Energy distribution in the LOUDEST frames ===")
        # Show the 5 loudest frames' per-channel breakdown.
        top_idx = sorted(loud_frames, key=lambda i: -frame_energies[i])[:5]
        for fi in top_idx:
            e = (frames[fi].samples ** 2).mean(axis=0)
            print(f"  frame {fi}: total={frame_energies[fi]:.3e}")
            for i, name in enumerate(names):
                if e[i] > 1e-7:
                    print(f"    {name}: {e[i]:.3e}")
    else:
        print("\n[diag] No loud frames detected. Possible causes:")
        print("  - No sound was actually played during the recording window.")
        print("  - Audio is routed to a different device (not VoiceMeeter Input).")
        print("  - The loopback stream is silent.")

    # ---- Diagnosis verdict ----
    print("\n=== VERDICT ===")
    surround_channels = [i for i, n in enumerate(names) if n in ("Ls", "Rs", "Lb", "Rb")]
    if surround_channels:
        surround_energy = per_ch_energy[surround_channels].max()
        front_energy = per_ch_energy[[names.index("L"), names.index("R")]].max()
        if front_energy > 1e-6 and surround_energy < front_energy * 0.05:
            print("⚠️  Surround channels (Ls/Rs/Lb/Rb) are essentially SILENT")
            print("    while L/R have signal. The game is NOT outputting real 7.1 —")
            print("    Windows up-mixed a stereo signal with empty surround channels.")
            print("    FIX: in the GAME's audio settings, select '7.1 Surround' /")
            print("    'Speakers' (NOT 'Headphones' / 'HRTF' / 'Stereo'), then restart the game.")
        elif front_energy < 1e-6:
            print("⚠️  Even L/R are silent. No audio reached the loopback stream.")
            print("    FIX: check Windows default output is VoiceMeeter Input,")
            print("    and that the game is actually playing sound.")
        else:
            print("✅  Surround channels have signal. The game IS outputting 7.1.")
            print("    The radar should work — if it doesn't, the issue is in")
            print("    onset/direction thresholds. Run with --debug on main.py.")
    else:
        print("(stereo layout — surround check skipped)")

    return 0


if __name__ == "__main__":
    sys.exit(main())