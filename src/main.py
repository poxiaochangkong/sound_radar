"""
Sound Radar entry point.

Usage:
    python src/main.py [--config config.yaml] [--profile profiles/csgo.yaml] [--debug]

Before opening the UI, we run a WASAPI preflight check: resolve the device,
verify it advertises enough output channels for the configured layout, and
print an actionable error if the Windows default device is not configured
for multichannel output. See docs/06_calibration.md section 4.

--debug mode prints real-time per-frame diagnostics to the terminal:
    flux, SNR per channel, onset yes/no, direction angle + contributing
channels. Use it to see why footsteps/gunshots do or don't trigger.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Make `src.*` imports work when running `python src/main.py` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from src.audio.capture import AudioCapture, AudioDeviceError
from src.config_loader import load_config
from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Sound Radar")
    p.add_argument("--config", default="config.yaml",
                   help="path to config.yaml")
    p.add_argument("--profile", default=None,
                   help="optional path to a per-game profile yaml")
    p.add_argument("--list-devices", action="store_true",
                   help="list loopback devices and exit")
    p.add_argument("--debug", action="store_true",
                   help="print real-time per-frame diagnostics to the terminal")
    return p.parse_args()


def _preflight_check(capture: AudioCapture) -> None:
    """Resolve the device and check its advertised channel count.

    Prints an actionable message and exits if the device can't supply the
    configured channel layout. Called BEFORE the Qt event loop starts.
    """
    try:
        dev = capture.preflight_device()
        print(f"[preflight] device: {dev}")
        print(f"[preflight] layout '{capture.channel_layout}' requires "
              f"{capture.expected_channels} channels — OK")
    except AudioDeviceError as e:
        print(f"[ERROR] Audio preflight failed:\n  {e}", file=sys.stderr)
        print("\nAvailable loopback devices:", file=sys.stderr)
        try:
            for d in AudioCapture.list_devices():
                print(f"  {d}", file=sys.stderr)
        except Exception:
            pass
        sys.exit(2)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.list_devices:
        print("Loopback devices:")
        for d in AudioCapture.list_devices():
            print(f"  {d}")
        return 0

    # Resolve config path relative to CWD.
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    try:
        config = load_config(config_path, profile_path=args.profile)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)

    win = MainWindow(config, debug=args.debug)

    try:
        _preflight_check(win._capture)
    except SystemExit:
        raise

    win.show()
    win.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())