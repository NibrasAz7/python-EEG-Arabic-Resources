"""Band-pass Butterworth filter for EEG signals.

Applies a 1-40 Hz band-pass filter to the P4 channel of the local
auditory EEG dataset in a single step and plots the result.

Usage:
    python bandpass.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 1000
LOWCUT = 1.0   # High-pass cutoff (Hz)
HIGHCUT = 40.0  # Low-pass cutoff (Hz)
ORDER = 4


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4
) -> np.ndarray:
    """Apply a band-pass Butterworth filter using filtfilt (zero-phase).

    Args:
        data: 1D EEG signal array.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band", analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=1, experiment=1, session=1
    )
    channel_data = eeg_data[:, 0]  # P4

    filtered = butter_bandpass_filter(
        channel_data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=ORDER
    )

    n_plot = 5000
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps[:n_plot], channel_data[:n_plot], label="Raw", color="gray")
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Raw EEG (P4)")
    axes[1].plot(timestamps[:n_plot], filtered[:n_plot], label="Band-pass", color="blue")
    axes[1].set_ylabel("EEG (uV)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title(f"Band-pass filtered ({LOWCUT}-{HIGHCUT} Hz)")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "bandpass_result.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
