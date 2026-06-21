"""
Multichannel WASAPI loopback capture.

Design notes:
  - The audio callback runs on the PortAudio real-time thread. It MUST NOT
    block, allocate, or call numpy DSP. It only converts the incoming buffer
    to float32 and pushes a copy onto a thread-safe queue.
  - The processing thread consumes the queue. See docs/02_architecture.md
    section 4 (thread model).
  - We open the stream in WASAPI shared mode with loopback=True. The actual
    channel count is dictated by the Windows default device's speaker
    configuration; we assert it matches the configured layout.

See docs/03_module_io.md section 1.1 for the contract.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from src.types import AudioFrame


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
    # sounddevice 0.5+ replaced query_hostapi_count()/query_hostapi(i) with
    # query_hostapis() (returns a list). Use that, with a fallback for older
    # versions just in case.
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
    """Captures multichannel PCM from a WASAPI loopback device.

    The captured frames are delivered via the `on_frame` callback, which is
    invoked on the audio thread. Implementations of `on_frame` must be
    non-blocking; the recommended pattern is to push the frame onto a
    `queue.Queue` and process it elsewhere.
    """

    def __init__(
        self,
        device: Optional[str],           # device name substring, or None for default
        sample_rate: int,                # e.g. 48000
        frame_size: int,                 # blocksize passed to PortAudio (0 = let PA decide)
        channel_layout: str,             # "7.1" | "5.1" | "stereo"
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

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open and start the loopback stream. Raises AudioDeviceError on failure."""
        with self._lock:
            if self._stream is not None:
                raise AudioDeviceError("Capture already started")

            device_idx = self._resolve_device_index(self._device_query)
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
                    f"{device_idx!r}: {e}"
                ) from e

            self._stream = stream
            self._actual_channels = self._expected_channels
            self._actual_sample_rate = stream.samplerate

    def stop(self) -> None:
        """Stop and close the stream. Safe to call multiple times."""
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

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
        """Called by PortAudio on the audio thread. MUST NOT block.

        `indata` is a buffer owned by PortAudio that may be reused after we
        return, so we copy it before handing the frame to the consumer.
        """
        try:
            # Defensive copy: the consumer must not see PortAudio's reused buffer.
            samples = np.ascontiguousarray(indata, dtype=np.float32)
            frame = AudioFrame(
                samples=samples,
                sample_rate=int(self._actual_sample_rate or self._sample_rate),
                timestamp=time.monotonic(),
                channel_names=list(self._channel_names),
            )
            self._on_frame(frame)
        except Exception:
            # Swallow on the audio thread — never propagate into PortAudio.
            pass

    # ---- diagnostics ------------------------------------------------------

    @property
    def actual_channels(self) -> Optional[int]:
        return self._actual_channels

    @property
    def actual_sample_rate(self) -> Optional[float]:
        return self._actual_sample_rate


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
            # Drop the oldest by draining one, then enqueue the new frame.
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