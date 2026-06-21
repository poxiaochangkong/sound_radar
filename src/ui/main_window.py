"""
Main application window.

Responsibilities:
  - Start / stop AudioCapture (WASAPI loopback)
  - Run a processing thread that consumes the frame queue and runs analysis
  - Drive the UI via a QTimer (~60 Hz): pull contacts, advance sweep
  - Expose a single Sensitivity slider that re-maps to SNR threshold and
    flux multiplier (see docs/06_calibration.md section 2)

Thread model:
  - Audio callback (PortAudio RT)  -> frame queue
  - Processing thread (daemon)     -> consumes frames, runs DSP, pushes contacts
  - UI thread (Qt)                 -> polls contact queue via QTimer

Concurrency contract (docs/03_module_io.md section 6):
  - The UI thread never mutates processing state directly.
  - Tunable parameters are read by the processing thread from a locked snapshot.
  - The onset detector's flux_multiplier is updated through its own locked setter.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QSlider,
                              QVBoxLayout, QWidget)

from src.analysis.direction import estimate_direction
from src.analysis.features import DEFAULT_BANDS, compute_band_energy, compute_channel_energy
from src.analysis.noise_floor import NoiseFloorEstimator
from src.analysis.onset import OnsetDetector
from src.audio.capture import AudioCapture, AudioDeviceError, make_drop_on_full_queue
from src.config_loader import AppConfig
from src.tracking.smoother import ContactSmoother
from src.models import AudioFrame
from src.ui.radar_widget import RadarWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._closing = False

        self.setWindowTitle(config.ui.window_title)
        self.resize(config.ui.width, config.ui.height)
        if config.ui.always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        # ---- UI layout ----
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self._radar = RadarWidget(colors=config.ui.radar)
        root.addWidget(self._radar, stretch=1)

        # Sensitivity slider (0..100, default 50).
        # HIGHER value = MORE sensitive = LOWER thresholds.
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Sensitivity"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setToolTip("0 = least sensitive (quiet rooms), 100 = most sensitive")
        self._slider.valueChanged.connect(self._on_sensitivity_changed)
        ctrl.addWidget(self._slider, stretch=1)
        self._status = QLabel("idle")
        ctrl.addWidget(self._status)
        root.addLayout(ctrl)

        self.setCentralWidget(central)

        # ---- Audio capture + queues ----
        self._frame_queue, self._on_frame_cb, self._dropped = make_drop_on_full_queue(maxsize=4)
        self._contact_queue: "queue.Queue[list]" = queue.Queue(maxsize=2)

        self._capture = AudioCapture(
            device=config.audio.device,
            sample_rate=config.audio.sample_rate,
            frame_size=config.audio.frame_size,
            channel_layout=config.audio.channel_layout,
            on_frame=self._on_frame_cb,
        )

        # ---- Processing state ----
        n_channels = len(config.channel_layout_obj.names)
        self._noise_floor = NoiseFloorEstimator(
            n_channels=n_channels,
            sample_rate=config.audio.sample_rate,
            frame_size=config.audio.frame_size,
        )
        self._onset = OnsetDetector(
            n_bands=len(DEFAULT_BANDS),
            flux_multiplier=config.onset.flux_multiplier,
            refractory_ms=config.onset.refractory_ms,
        )
        self._smoother = ContactSmoother(
            decay_ms=config.tracking.decay_ms,
            angle_smoothing=config.tracking.angle_smoothing,
        )

        # Locked snapshot of tunable parameters shared with the processing thread.
        self._params_lock = threading.Lock()
        self._params = {
            "snr_threshold_db": float(config.direction.snr_threshold_db),
            "ignore_channels": (["C", "LFE"]
                                 if config.direction.ignore_center_channel else []),
        }

        self._proc_thread: Optional[threading.Thread] = None

        # ---- UI timer (~60 Hz): poll contacts + advance sweep ----
        self._last_tick = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(16)   # ~60 Hz
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # ---- sensitivity slider ----------------------------------------------

    def _on_sensitivity_changed(self, value: int) -> None:
        # HIGHER slider value = MORE sensitive.
        # Map slider 0..100 to sensitivity_factor 0.5..2.0 (higher = more sensitive).
        # Effective threshold = base / factor -> higher factor -> lower threshold -> more sensitive.
        norm = value / 100.0
        sensitivity_factor = 0.5 + 1.5 * norm   # 0.5 at slider=0, 2.0 at slider=100

        base_snr = float(self._config.direction.snr_threshold_db)
        base_flux = float(self._config.onset.flux_multiplier)
        new_snr = max(1.0, base_snr / sensitivity_factor)
        new_flux = max(0.5, base_flux / sensitivity_factor)

        # Update SNR threshold via the locked snapshot (processing thread reads it).
        with self._params_lock:
            self._params["snr_threshold_db"] = new_snr

        # Update flux multiplier via the onset detector's own locked setter,
        # respecting the concurrency contract (no direct attribute writes).
        self._onset.set_flux_multiplier(new_flux)

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        try:
            self._capture.start()
        except AudioDeviceError as e:
            logger.error("audio device error: %s", e)
            self._status.setText(f"device error: {e}")
            return
        self._proc_thread = threading.Thread(target=self._processing_loop,
                                             daemon=True)
        self._proc_thread.start()
        ch = self._capture.actual_channels
        sr = self._capture.actual_sample_rate
        self._status.setText(f"running: {ch}ch @ {int(sr)}Hz")
        logger.info("audio started: %sch @ %sHz", ch, sr)

    def closeEvent(self, event) -> None:
        self._closing = True
        try:
            self._capture.stop()
        except Exception as e:
            logger.warning("error stopping capture: %s", e)
        super().closeEvent(event)

    # ---- processing thread -----------------------------------------------

    def _processing_loop(self) -> None:
        cfg = self._config
        layout = cfg.channel_layout_obj
        sample_rate = cfg.audio.sample_rate

        while not self._closing:
            try:
                frame: AudioFrame = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                # No audio: still decay existing contacts so they fade out.
                contacts = self._smoother.update(None)
                self._push_contacts(contacts)
                continue

            try:
                contacts = self._process_one_frame(frame, layout, sample_rate)
                self._push_contacts(contacts)
            except Exception as e:
                # Processing must never die; log and keep going.
                logger.exception("error in processing loop: %s", e)

    def _process_one_frame(self, frame: AudioFrame, layout, sample_rate: int):
        # Snapshot of live params.
        with self._params_lock:
            snr_threshold = self._params["snr_threshold_db"]
            ignore_channels = list(self._params["ignore_channels"])

        # 1. Features.
        channel_energy = compute_channel_energy(frame.samples)
        band_energy = compute_band_energy(frame.samples, sample_rate)

        # 2. Onset detection FIRST (before noise floor update), so that if an
        #    onset is detected we can FREEZE the noise floor for this frame
        #    and prevent the transient from polluting the estimate.
        onset_event = self._onset.process(frame, band_energy)
        if onset_event is not None:
            # Skip the noise-floor update for this transient frame.
            self._noise_floor.freeze()
        noise_floor = self._noise_floor.update(channel_energy)

        # 3. Direction estimation (only on onset frames).
        estimate = None
        if onset_event is not None:
            estimate = estimate_direction(
                channel_energy=channel_energy,
                layout=layout,
                noise_floor=noise_floor,
                snr_threshold_db=snr_threshold,
                ignore_channels=ignore_channels,
                timestamp=onset_event.timestamp,
            )

        # 4. Smoother (decay + merge/create).
        return self._smoother.update(estimate)

    def _push_contacts(self, contacts) -> None:
        try:
            try:
                self._contact_queue.put_nowait(list(contacts))
            except queue.Full:
                try:
                    self._contact_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._contact_queue.put_nowait(list(contacts))
                except queue.Full:
                    pass
        except Exception as e:
            logger.warning("error pushing contacts: %s", e)

    # ---- UI tick ---------------------------------------------------------

    def _on_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # Drain the contact queue (keep latest).
        latest = None
        while True:
            try:
                latest = self._contact_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._radar.set_contacts(latest)

        # Advance sweep animation.
        self._radar.advance_sweep(dt)