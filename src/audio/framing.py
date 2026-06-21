"""
Ring buffer and windowing utilities for spectral analysis.

See docs/03_module_io.md section 1.2 for the contract.
"""
from typing import Optional

import numpy as np


class RingBuffer:
    """Fixed-capacity ring buffer for multichannel PCM samples.

    Holds the most recent `capacity_samples` samples across `n_channels`
    channels. When the buffer has not yet been filled, `read_window` returns
    zero-padded data (oldest positions are zero).

    Thread-safety: NOT thread-safe. The buffer is owned by a single thread
    (the processing thread) and is never touched from the audio callback.
    """

    def __init__(self, n_channels: int, capacity_samples: int) -> None:
        if n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {n_channels}")
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be > 0, got {capacity_samples}")
        self._n_channels = int(n_channels)
        self._capacity = int(capacity_samples)
        # Pre-allocate buffer as zeros so unfilled positions are zero by default.
        self._buf = np.zeros((self._capacity, self._n_channels), dtype=np.float32)
        self._write_pos = 0       # next write index in [0, capacity)
        self._filled = 0          # how many valid samples have been written, capped at capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def push(self, samples: np.ndarray) -> None:
        """Append samples (shape=(N, n_channels), float32) to the buffer.

        If N >= capacity, only the last `capacity` samples are kept.
        """
        if samples.ndim != 2 or samples.shape[1] != self._n_channels:
            raise ValueError(
                f"samples must have shape (N, {self._n_channels}), got {samples.shape}"
            )
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        n = samples.shape[0]
        if n == 0:
            return

        # If the push is larger than the buffer, keep only the tail.
        if n >= self._capacity:
            self._buf[:] = samples[-self._capacity:]
            self._write_pos = 0
            self._filled = self._capacity
            return

        # General case: copy in up to two chunks (ring wrap).
        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = samples
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:first]
            self._buf[:end - self._capacity] = samples[first:]
        self._write_pos = end % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def read_window(self, size: int) -> np.ndarray:
        """Return the most recent `size` samples, oldest first.

        If fewer than `size` samples have been pushed, the leading positions
        are zero (zero-padded). The returned array is always shape
        (size, n_channels) and float32.
        """
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if size > self._capacity:
            raise ValueError(
                f"size {size} exceeds buffer capacity {self._capacity}"
            )
        out = np.zeros((size, self._n_channels), dtype=np.float32)
        # How many valid samples can we actually provide?
        available = min(self._filled, size)
        if available == 0:
            return out
        # The most recent `available` samples live just before write_pos.
        # Compute their start index in the circular buffer.
        start = (self._write_pos - available) % self._capacity
        # Place them at the tail of `out` so leading positions stay zero.
        out_offset = size - available
        if start + available <= self._capacity:
            out[out_offset:] = self._buf[start:start + available]
        else:
            first = self._capacity - start
            out[out_offset:out_offset + first] = self._buf[start:]
            out[out_offset + first:] = self._buf[:available - first]
        return out

    def reset(self) -> None:
        """Clear the buffer back to the all-zero state."""
        self._buf.fill(0)
        self._write_pos = 0
        self._filled = 0


def hann_window(size: int) -> np.ndarray:
    """Return a Hann window of the given size. Output range [0, 1]."""
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if size == 1:
        return np.ones(1, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / (size - 1))