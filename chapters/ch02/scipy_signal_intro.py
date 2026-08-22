"""Introduction to SciPy signal processing for EEG.

Demonstrates designing a bandpass filter (8-13 Hz, alpha band)
and applying it to a test signal using zero-phase filtering.

Usage:
    python scipy_signal_intro.py
"""

import numpy as np
from scipy import signal


def main() -> None:
    fs = 200  # Sampling rate (Hz)

    # Design a bandpass filter (8-13 Hz, alpha band)
    # 4th order Butterworth filter
    low, high = 8.0, 13.0
    nyquist = fs / 2.0  # Nyquist frequency
    b, a = signal.butter(4, [low / nyquist, high / nyquist], btype='band')

    # Generate a test signal: 10 Hz sine + noise
    np.random.seed(42)
    t = np.arange(1000) / fs
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(1000)

    # Apply zero-phase filter (filtfilt)
    filtered = signal.filtfilt(b, a, test_signal)

    print(f"Original signal std: {test_signal.std():.3f}")
    print(f"Filtered signal std: {filtered.std():.3f}")

    # Compute power spectral density to verify band isolation
    freqs, psd = signal.welch(filtered, fs=fs, nperseg=256)
    peak_idx = np.argmax(psd)
    print(f"Peak frequency: {freqs[peak_idx]:.1f} Hz")
    print(f"Expected: 10.0 Hz (alpha band center)")


if __name__ == "__main__":
    main()
