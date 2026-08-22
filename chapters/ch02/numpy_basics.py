"""Basic NumPy operations for EEG data.

Demonstrates array creation, shape inspection, vectorized operations,
and baseline removal on simulated 4-channel EEG data.

Usage:
    python numpy_basics.py
"""

import numpy as np


def main() -> None:
    # Create a 2D array: 4 channels x 1000 time points
    # Simulating 5 seconds of EEG at 200 Hz sampling rate
    np.random.seed(42)
    eeg_data = np.random.randn(4, 1000) * 50  # 50 microvolts scale

    # Array shape: (channels, samples)
    print(f"Shape: {eeg_data.shape}")
    print(f"Channels: {eeg_data.shape[0]}")
    print(f"Samples per channel: {eeg_data.shape[1]}")

    # Vectorized operations (no loops needed)
    # Subtract the mean of each channel (baseline removal)
    eeg_centered = eeg_data - eeg_data.mean(axis=1, keepdims=True)

    # Compute amplitude range for each channel
    amplitude = eeg_data.max(axis=1) - eeg_data.min(axis=1)
    print(f"Amplitude range per channel: {amplitude}")

    # Verify baseline removal
    print(f"Mean before removal: {eeg_data.mean(axis=1)}")
    print(f"Mean after removal:  {eeg_centered.mean(axis=1)}")


if __name__ == "__main__":
    main()
