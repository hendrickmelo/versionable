"""Simple DAQ simulator for the save-as-you-go example.

This is demo/helper code — not part of the versionable library.
It generates a sinusoidal signal in chunks, simulating a real
data acquisition device.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class DaqSimulator:
    """Generates a sinusoidal signal in chunks."""

    def __init__(
        self,
        frequency_Hz: float = 10.0,
        amplitude_V: float = 1.0,
        sampleRate_Hz: float = 1000.0,
        chunkSize: int = 100,
        duration_s: float = 1.0,
    ) -> None:
        self.frequency_Hz = frequency_Hz
        self.amplitude_V = amplitude_V
        self.sampleRate_Hz = sampleRate_Hz
        self.chunkSize = chunkSize
        self.duration_s = duration_s

    def stream(self, startOffset_s: float = 0.0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (time_chunk, voltage_chunk) tuples."""
        dt = 1.0 / self.sampleRate_Hz
        totalSamples = int(self.duration_s * self.sampleRate_Hz)
        sampleIdx = 0

        while sampleIdx < totalSamples:
            n = min(self.chunkSize, totalSamples - sampleIdx)
            t = startOffset_s + (sampleIdx + np.arange(n)) * dt
            v = self.amplitude_V * np.sin(2 * np.pi * self.frequency_Hz * t)
            yield t, v
            sampleIdx += n
