"""High-pass Butterworth filter for EEG signals.

Applies a 1 Hz high-pass filter to the P4 channel of the local
auditory EEG dataset and plots the result before and after filtering.

Usage:
    python highpass.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 1000  # Sampling rate (Hz)
CUTOFF = 1.0  # High-pass cutoff (Hz)
ORDER = 4  # Butterworth filter order


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a high-pass Butterworth filter using filtfilt (zero-phase).

    Args:
        data: 1D EEG signal array.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=1, experiment=1, session=1
    )
    channel_data = eeg_data[:, 0]  # P4

    filtered = butter_highpass_filter(channel_data, cutoff=CUTOFF, fs=FS, order=ORDER)

    n_plot = 5000
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps[:n_plot], channel_data[:n_plot], label="Raw")
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Before high-pass filter")
    axes[1].plot(timestamps[:n_plot], filtered[:n_plot], label="Filtered", color="green")
    axes[1].set_ylabel("EEG (uV)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title(f"After high-pass filter ({CUTOFF} Hz)")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "highpass_result.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
