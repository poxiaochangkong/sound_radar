"""
Sound Radar entry point.

Usage:
    python src/main.py [--config config.yaml] [--profile profiles/csgo.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `src.*` imports work when running `python src/main.py` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from src.config_loader import load_config
from src.ui.main_window import MainWindow


def parse_args():
    p = argparse.ArgumentParser(description="Sound Radar")
    p.add_argument("--config", default="config.yaml",
                   help="path to config.yaml")
    p.add_argument("--profile", default=None,
                   help="optional path to a per-game profile yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve config path relative to CWD.
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # Fall back to a project-root-relative path.
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
    win = MainWindow(config)
    win.show()
    win.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())