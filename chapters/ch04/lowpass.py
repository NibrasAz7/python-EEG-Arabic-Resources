"""Low-pass Butterworth filter for EEG signals.

Applies a 40 Hz low-pass filter to the output of the high-pass filter
and plots the result before and after.

Usage:
    python lowpass.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from utils.eeg_loader import load_local_eeg
from chapters.ch04.highpass import butter_highpass_filter

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
CUTOFF = 40.0  # Low-pass cutoff (Hz)
ORDER = 4


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a low-pass Butterworth filter using filtfilt (zero-phase).

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
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]  # P4

    # First apply high-pass, then low-pass
    filtered_hp = butter_highpass_filter(channel_data, cutoff=1.0, fs=FS, order=ORDER)
    filtered_lp = butter_lowpass_filter(filtered_hp, cutoff=CUTOFF, fs=FS, order=ORDER)

    n_plot = 5000
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps[:n_plot], filtered_hp[:n_plot], label="After HP only")
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Before low-pass filter")
    axes[1].plot(timestamps[:n_plot], filtered_lp[:n_plot], label="After LP", color="red")
    axes[1].set_ylabel("EEG (uV)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title(f"After low-pass filter ({CUTOFF} Hz)")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "lowpass_result.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
