"""
Multichannel WASAPI loopback capture via PyAudioWPatch.

Why PyAudioWPatch (not `sounddevice`):
  `sounddevice` >= 0.5 removed WASAPI loopback support
  (`WasapiSettings(loopback=True)` was deleted). PyAudioWPatch (a PyAudio
  fork) instead exposes each render device's loopback as a separate *virtual
  input device* whose name ends with "[Loopback]". We open that virtual
  input device like a normal InputStream, and we get the device's output
  mix as PCM. Stable across versions.

Note on host API:
  PyAudioWPatch registers the loopback virtual devices under the MME host
  API (hostApi id == paMME == 2), NOT under paWASAPI (13). This is fine —
  PortAudio still talks to WASAPI under the hood. We therefore do NOT
  filter loopback devices by host API; we filter by `isLoopbackDevice`.

Design notes:
  - The audio callback runs on PortAudio's real-time thread. It MUST NOT
    block, allocate, or call numpy DSP. It only copies the incoming buffer
    to float32 and pushes a snapshot onto a thread-safe queue.
  - The processing thread consumes the queue (see docs/02_architecture.md
    section 4, thread model).
  - On start() we verify that the loopback device advertises enough input
    channels for the configured layout and raise a clear error if not
    (docs/06_calibration.md section 4 — multichannel self-check).

Public API is identical to the previous sounddevice-based implementation:
  AudioCapture(device, sample_rate, frame_size, channel_layout, on_frame)
    .start() / .stop() / .preflight_device()
     .channel_names / .actual_channels / .actual_sample_rate
  AudioCapture.list_devices()
  make_drop_on_full_queue(maxsize) -> (queue, put_fn, dropped_count_fn)

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
import pyaudiowpatch as pyaudio

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
    max_output_channels: int   # for loopback devices, this holds maxInputChannels
    default_sample_rate: float
    is_loopback: bool = False

    def __str__(self) -> str:
        # PyAudioWPatch already appends " [Loopback]" to the device name, so
        # don't append it again in the display string.
        return (f"[{self.index}] {self.name} "
                f"({self.host_api}, in={self.max_output_channels}ch)")


def _pyaudio_host_api_name(host_api_type_id: int) -> str:
    """Map PortAudio host-api type id to a friendly name."""
    mapping = {
        pyaudio.paMME: "MME",
        pyaudio.paDirectSound: "Windows DirectSound",
        pyaudio.paWASAPI: "Windows WASAPI",
        pyaudio.paWDMKS: "Windows WDM-KS",
    }
    return mapping.get(host_api_type_id, f"api#{host_api_type_id}")


def _info_to_loopback_device_info(info: dict) -> DeviceInfo:
    """Build a DeviceInfo from a PyAudioWPatch loopback device dict."""
    return DeviceInfo(
        index=int(info["index"]),
        name=str(info["name"]),
        host_api=_pyaudio_host_api_name(int(info.get("hostApi", -1))),
        # For loopback devices, the readable channel count is maxInputChannels.
        max_output_channels=int(info["maxInputChannels"]),
        default_sample_rate=float(info["defaultSampleRate"]),
        is_loopback=True,
    )


class AudioCapture:
    """Captures multichannel PCM from a WASAPI loopback device.

    Public contract is identical to the previous sounddevice-based version.
    """

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

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._actual_channels: Optional[int] = None
        self._actual_sample_rate: Optional[float] = None
        self._lock = threading.Lock()

    # ---- introspection ----------------------------------------------------

    @staticmethod
    def list_devices(host_api: Optional[str] = None) -> List[DeviceInfo]:
        """List loopback devices.

        `host_api` is accepted for backward compatibility but is NOT used to
        filter, because PyAudioWPatch registers loopback devices under MME
        regardless of the underlying API. We filter purely by
        `isLoopbackDevice`.
        """
        pa = pyaudio.PyAudio()
        try:
            out: List[DeviceInfo] = []
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if not info.get("isLoopbackDevice", False):
                    continue
                out.append(_info_to_loopback_device_info(info))
            return out
        finally:
            pa.terminate()

    def preflight_device(self) -> DeviceInfo:
        """Resolve and return the loopback device WITHOUT opening a stream.

        Raises AudioDeviceError if the device cannot be resolved or its
        loopback input channel count is below the configured layout.
        """
        pa = pyaudio.PyAudio()
        try:
            dev_idx, info = self._resolve_loopback_device(pa, self._device_query)
        finally:
            pa.terminate()
        dev = _info_to_loopback_device_info(info)
        if dev.max_output_channels < self._expected_channels:
            raise AudioDeviceError(
                f"Loopback device '{dev.name}' advertises only "
                f"{dev.max_output_channels} input channels, but layout "
                f"'{self._channel_layout}' requires {self._expected_channels}. "
                f"Configure the Windows default device to {self._channel_layout} "
                f"(or use a virtual 7.1 device like VoiceMeeter)."
            )
        return dev

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open and start the loopback stream. Raises AudioDeviceError on failure.

        Performs the multichannel self-check from docs/06_calibration.md section 4.
        """
        with self._lock:
            if self._stream is not None:
                raise AudioDeviceError("Capture already started")

            self._pa = pyaudio.PyAudio()
            try:
                dev_idx, dev_info = self._resolve_loopback_device(self._pa, self._device_query)
            except AudioDeviceError:
                self._pa.terminate()
                self._pa = None
                raise

            ch_in = int(dev_info["maxInputChannels"])
            if ch_in < self._expected_channels:
                self._pa.terminate()
                self._pa = None
                raise AudioDeviceError(
                    f"Loopback device '{dev_info['name']}' advertises only "
                    f"{ch_in} input channels, but layout "
                    f"'{self._channel_layout}' requires "
                    f"{self._expected_channels}. Configure the Windows default "
                    f"device to {self._channel_layout} (Sound Settings → "
                    f"Device properties → Additional device properties → "
                    f"Configure → 7.1 Surround)."
                )

            device_rate = int(dev_info["defaultSampleRate"])
            try:
                stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=self._expected_channels,
                    rate=device_rate,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=self._frame_size,
                    stream_callback=self._portaudio_callback,
                )
                stream.start_stream()
            except Exception as e:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
                raise AudioDeviceError(
                    f"Failed to open loopback stream on device "
                    f"{dev_idx!r} ({dev_info['name']}): {e}\n"
                    f"Hint: confirm the loopback device is a "
                    f"{self._channel_layout} output."
                ) from e

            self._stream = stream
            self._actual_channels = self._expected_channels
            self._actual_sample_rate = float(device_rate)
            logger.info(
                "WASAPI loopback opened: device='%s' %sch @ %sHz",
                dev_info["name"], self._actual_channels, self._actual_sample_rate,
            )

    def stop(self) -> None:
        """Stop and close the stream. Safe to call multiple times."""
        with self._lock:
            stream = self._stream
            pa = self._pa
            self._stream = None
            self._pa = None
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception as e:
                logger.warning("error stopping stream: %s", e)
            try:
                stream.close()
            except Exception as e:
                logger.warning("error closing stream: %s", e)
        if pa is not None:
            try:
                pa.terminate()
            except Exception as e:
                logger.warning("error terminating PyAudio: %s", e)

    # ---- helpers ----------------------------------------------------------

    def _resolve_loopback_device(self, pa: pyaudio.PyAudio,
                                  query: Optional[str]) -> tuple:
        """Find a loopback device matching `query`.

        If `query` is None, returns the default render device's loopback
        (via PyAudioWPatch's get_default_wasapi_loopback()).
        Otherwise matches by case-insensitive substring against the loopback
        device name (which includes the [Loopback] suffix).
        """
        if query is None:
            try:
                default_loop = pa.get_default_wasapi_loopback()
            except Exception as e:
                raise AudioDeviceError(
                    f"No default WASAPI loopback device available: {e}"
                )
            return int(default_loop["index"]), default_loop

        q = query.lower()
        # Match by substring against loopback device names.
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if not info.get("isLoopbackDevice", False):
                continue
            name = str(info["name"]).lower()
            if q in name:
                return int(i), info

        # Helpful error: show available loopback devices.
        names = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if info.get("isLoopbackDevice", False):
                names.append(f"{info['name']} (in={info['maxInputChannels']}ch)")
        raise AudioDeviceError(
            f"No loopback device name matches '{query}'. "
            f"Available loopback devices: " + ", ".join(names)
        )

    # ---- real-time callback ----------------------------------------------

    def _portaudio_callback(self, in_data: bytes, frame_count: int,
                             time_info, status) -> tuple:
        """Called by PortAudio on the audio thread. MUST NOT block."""
        try:
            # in_data is interleaved float32 bytes; reshape without copy where
            # possible. ascontiguousarray + frombuffer keeps it cheap.
            samples = np.frombuffer(in_data, dtype=np.float32).reshape(
                frame_count, self._expected_channels
            )
            # Defensive copy: PortAudio reuses the buffer after we return.
            samples = np.ascontiguousarray(samples, dtype=np.float32)
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
        return (None, pyaudio.paContinue)

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

    @property
    def expected_channels(self) -> int:
        """Number of channels required by the configured layout (for diagnostics)."""
        return self._expected_channels

    @property
    def channel_layout(self) -> str:
        return self._channel_layout


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