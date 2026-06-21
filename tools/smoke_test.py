"""
Smoke test: import every module and briefly instantiate the UI.

Runs headless-safe: creates the QApplication, shows the window, and quits
after ~2 seconds via a one-shot QTimer. This does NOT open the audio device;
it only verifies that the whole stack imports and constructs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src.config_loader import load_config
from src.ui.main_window import MainWindow
from src.types import RadarContact


def main() -> int:
    # 1. Import-check every module explicitly.
    import src.types
    import src.config_loader
    import src.audio.capture
    import src.audio.framing
    import src.analysis.features
    import src.analysis.noise_floor
    import src.analysis.onset
    import src.analysis.direction
    import src.tracking.smoother
    import src.ui.radar_widget
    import src.ui.main_window
    print("[smoke] all modules imported OK")

    # 2. Load config.
    cfg = load_config("config.yaml")
    print(f"[smoke] config loaded: layout={cfg.audio.channel_layout} "
          f"channels={len(cfg.channel_layout_obj.names)}")

    # 3. Build the app + window.
    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.show()

    # 4. Inject a fake contact to verify painting path works.
    win._radar.set_contacts([
        RadarContact(angle_deg=-30.0, intensity=1.0, born_at=0.0),
        RadarContact(angle_deg=110.0, intensity=0.6, born_at=0.0),
    ])

    # 5. Quit after 2 seconds (one-shot timer).
    QTimer.singleShot(2000, app.quit)
    print("[smoke] window shown, quitting in 2s...")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())