"""
Record a few seconds of multichannel loopback audio to a WAV file.

Useful for verifying that the Windows default device is actually outputting
the expected number of channels (see docs/06_calibration.md section 5.3).

Usage:
    python tools/record_session.py --seconds 5 --out recordings/test.wav
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.capture import AudioCapture, AudioDeviceError, make_drop_on_full_queue
from src.config_loader import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Record multichannel loopback")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--out", default="recordings/session.wav")
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

    print(f"[record] opening device (layout={cfg.audio.channel_layout})...")
    try:
        capture.start()
    except AudioDeviceError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"[record] capturing {args.seconds:.1f}s ...")
    deadline = time.monotonic() + args.seconds
    chunks = []
    while time.monotonic() < deadline:
        try:
            chunks.append(frame_queue.get(timeout=0.1).samples)
        except queue.Empty:
            continue
    capture.stop()

    if not chunks:
        print("[record] no audio captured", file=sys.stderr)
        return 1

    audio = np.concatenate(chunks, axis=0)
    print(f"[record] captured shape={audio.shape} dtype={audio.dtype}")

    # Save as 32-bit float WAV using scipy.
    from scipy.io import wavfile
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    wavfile.write(args.out, int(cfg.audio.sample_rate),
                  audio.astype(np.float32))
    print(f"[record] wrote {args.out} "
          f"({audio.shape[1]} channels, {audio.shape[0]/cfg.audio.sample_rate:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())