"""Introduction to MNE-Python for EEG data.

Demonstrates creating an MNE Raw object from synthetic data,
inspecting channel info, and basic properties.

Usage:
    python mne_intro.py
"""

import numpy as np
import mne


def main() -> None:
    # Create a synthetic EEG signal (4 channels, 10 seconds, 200 Hz)
    fs = 200
    n_samples = fs * 10
    np.random.seed(42)
    data = np.random.randn(4, n_samples) * 50  # 50 uV scale

    # Create MNE Raw object
    info = mne.create_info(
        ch_names=['P4', 'Cz', 'F8', 'T7'],
        sfreq=fs,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info)

    print(raw)
    print(f"Channels: {raw.ch_names}")
    print(f"Duration: {raw.times[-1]:.1f} s")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    # Get data as numpy array
    data_out, times = raw.get_data(return_times=True)
    print(f"Data shape: {data_out.shape}")
    print(f"Times shape: {times.shape}")


if __name__ == "__main__":
    main()
