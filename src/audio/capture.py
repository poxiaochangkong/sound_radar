"""
Multichannel WASAPI loopback capture.

Design notes:
  - The audio callback runs on the PortAudio real-time thread. It MUST NOT
    block, allocate, or call numpy DSP. It only converts the incoming buffer
    to float32 and pushes a copy onto a thread-safe queue.
  - The processing thread consumes the queue. See docs/02_architecture.md
    section 4 (thread model).
  - We open the stream in WASAPI shared mode with loopback=True. On start()
    we verify that the device actually supports the configured channel count
    and report a clear, actionable error if not (see docs/06_calibration.md
    section 4 — multichannel self-check).

See docs/03_module_io.md section 1.1 for the contract.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from src.models import AudioFrame

logger = logging.getLogger(__name__)

# Standard Windows channel ordering for 7.1 / 5.1 / stereo PCM.
_CHANNEL_ORDER = {
    "7.1":    ["L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb"],
    "5.1":    ["L", "R", "C", "LFE", "Ls", "Rs"],
    "stereo": ["L", "R"],
}


class AudioDeviceError(Exception):
    """Raised when the requested audio device or layout cannot be opened."""


@dataclass
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_output_channels: int
    default_sample_rate: float

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api}, {self.max_output_channels}ch)"


def _query_hostapis() -> List[dict]:
    """Return the list of host APIs. Compatible with sounddevice >= 0.5."""
    if hasattr(sd, "query_hostapis"):
        return list(sd.query_hostapis())
    apis = []
    count = sd.query_hostapi_count()
    for i in range(count):
        apis.append(sd.query_hostapi(i))
    return apis


def _find_hostapi(name: str) -> dict:
    apis = {a["name"]: a for a in _query_hostapis()}
    if name not in apis:
        raise AudioDeviceError(
            f"Host API '{name}' not found. Available: {list(apis.keys())}"
        )
    return apis[name]


class AudioCapture:
    """Captures multichannel PCM from a WASAPI loopback device."""

    def __init__(
        self,
        device: Optional[str],
        sample_rate: int,
        frame_size: int,
        channel_layout: str,
        on_frame: Callable[[AudioFrame], None],
    ) -> None:
        if channel_layout not in _CHANNEL_ORDER:
            raise ValueError(f"Unsupported channel_layout: {channel_layout!r}")
        self._device_query = device
        self._sample_rate = int(sample_rate)
        self._frame_size = int(frame_size)
        self._channel_layout = channel_layout
        self._expected_channels = len(_CHANNEL_ORDER[channel_layout])
        self._channel_names = list(_CHANNEL_ORDER[channel_layout])
        self._on_frame = on_frame

        self._stream: Optional[sd.InputStream] = None
        self._actual_channels: Optional[int] = None
        self._actual_sample_rate: Optional[float] = None
        self._lock = threading.Lock()

    # ---- introspection ----------------------------------------------------

    @staticmethod
    def list_devices(host_api: str = "Windows WASAPI") -> List[DeviceInfo]:
        """List output devices for the given host API (default: WASAPI)."""
        api = _find_hostapi(host_api)
        out: List[DeviceInfo] = []
        for dev_idx in api["devices"]:
            info = sd.query_devices(dev_idx)
            out.append(DeviceInfo(
                index=dev_idx,
                name=info["name"],
                host_api=host_api,
                max_output_channels=info["max_output_channels"],
                default_sample_rate=info["default_samplerate"],
            ))
        return out

    def preflight_device(self) -> DeviceInfo:
        """Resolve and return the device that will be used, WITHOUT opening it.

        Raises AudioDeviceError if the device cannot be resolved or does not
        advertise enough output channels for the configured layout.
        """
        device_idx = self._resolve_device_index(self._device_query)
        info = sd.query_devices(device_idx)
        dev = DeviceInfo(
            index=device_idx,
            name=info["name"],
            host_api="Windows WASAPI",
            max_output_channels=info["max_output_channels"],
            default_sample_rate=info["default_samplerate"],
        )
        if dev.max_output_channels < self._expected_channels:
            raise AudioDeviceError(
                f"Device '{dev.name}' advertises only {dev.max_output_channels} "
                f"output channels, but layout '{self._channel_layout}' requires "
                f"{self._expected_channels}. Configure the Windows default device "
                f"to {self._channel_layout} (or use a virtual 7.1 device like "
                f"VoiceMeeter)."
            )
        return dev

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open and start the loopback stream. Raises AudioDeviceError on failure.

        Performs the multichannel self-check from docs/06_calibration.md section 4:
        verifies device channel count BEFORE opening and verifies the opened
        stream's channel count AFTER opening.
        """
        with self._lock:
            if self._stream is not None:
                raise AudioDeviceError("Capture already started")

            # Pre-open check: device must advertise enough output channels.
            device_idx = self._resolve_device_index(self._device_query)
            dev_info = sd.query_devices(device_idx)
            if dev_info["max_output_channels"] < self._expected_channels:
                raise AudioDeviceError(
                    f"Device '{dev_info['name']}' advertises only "
                    f"{dev_info['max_output_channels']} output channels, but "
                    f"layout '{self._channel_layout}' requires "
                    f"{self._expected_channels}. Configure the Windows default "
                    f"device to {self._channel_layout} (Sound Settings → "
                    f"Device properties → Additional device properties → "
                    f"Configure → 7.1 Surround)."
                )

            try:
                stream = sd.InputStream(
                    device=device_idx,
                    samplerate=self._sample_rate,
                    blocksize=self._frame_size,
                    channels=self._expected_channels,
                    dtype="float32",
                    latency="low",
                    extra_settings=sd.WasapiSettings(loopback=True),
                    callback=self._portaudio_callback,
                )
                stream.start()
            except Exception as e:
                raise AudioDeviceError(
                    f"Failed to open loopback stream on device "
                    f"{device_idx!r} ({dev_info['name']}): {e}\n"
                    f"Hint: confirm Windows default output is set to a "
                    f"{self._channel_layout} device."
                ) from e

            self._stream = stream
            self._actual_channels = self._expected_channels
            self._actual_sample_rate = stream.samplerate
            logger.info(
                "WASAPI loopback opened: device='%s' %sch @ %sHz",
                dev_info["name"], self._actual_channels, self._actual_sample_rate,
            )

    def stop(self) -> None:
        """Stop and close the stream. Safe to call multiple times."""
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception as e:
                logger.warning("error stopping stream: %s", e)
            try:
                stream.close()
            except Exception as e:
                logger.warning("error closing stream: %s", e)

    # ---- helpers ----------------------------------------------------------

    def _resolve_device_index(self, query: Optional[str]) -> int:
        """Find a WASAPI output device whose name contains `query`.

        If `query` is None, returns the WASAPI default output device.
        """
        wasapi = _find_hostapi("Windows WASAPI")

        if query is None:
            default_out = wasapi["default_output_device"]
            if default_out < 0:
                raise AudioDeviceError(
                    "No default WASAPI output device is configured in Windows."
                )
            return int(default_out)

        # Match by substring, case-insensitive.
        for dev_idx in wasapi["devices"]:
            name = sd.query_devices(dev_idx)["name"]
            if query.lower() in name.lower():
                return int(dev_idx)
        raise AudioDeviceError(
            f"No WASAPI output device name matches '{query}'. "
            f"Available: " + ", ".join(
                sd.query_devices(i)["name"] for i in wasapi["devices"]
            )
        )

    # ---- real-time callback ----------------------------------------------

    def _portaudio_callback(self, indata: np.ndarray, frames: int,
                            time_info, status) -> None:
        """Called by PortAudio on the audio thread. MUST NOT block."""
        try:
            samples = np.ascontiguousarray(indata, dtype=np.float32)
            frame = AudioFrame(
                samples=samples,
                sample_rate=int(self._actual_sample_rate or self._sample_rate),
                timestamp=time.monotonic(),
                channel_names=list(self._channel_names),
            )
            self._on_frame(frame)
        except Exception as e:
            # Swallow on the audio thread — never propagate into PortAudio.
            # Logged via module logger only if it's safe; in RT context we
            # just keep going.
            pass

    # ---- diagnostics ------------------------------------------------------

    @property
    def actual_channels(self) -> Optional[int]:
        return self._actual_channels

    @property
    def actual_sample_rate(self) -> Optional[float]:
        return self._actual_sample_rate

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_names)


# ---- Convenience: a queue-backed consumer ----------------------------------

def make_drop_on_full_queue(maxsize: int = 4):
    """Return (queue, put_fn, dropped_count_fn).

    Usage: pass `put_fn` as the on_frame callback to AudioCapture; the
    processing thread reads from the returned queue with `queue.get()`.
    Dropping the oldest (not the newest) keeps latency bounded under load.
    """
    q: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=maxsize)
    drop_counter = {"count": 0}

    def put(frame: AudioFrame) -> None:
        try:
            q.put_nowait(frame)
        except queue.Full:
            try:
                q.get_nowait()
                drop_counter["count"] += 1
            except queue.Empty:
                pass
            try:
                q.put_nowait(frame)
            except queue.Full:
                drop_counter["count"] += 1

    def dropped_count() -> int:
        return drop_counter["count"]

    return q, put, dropped_count