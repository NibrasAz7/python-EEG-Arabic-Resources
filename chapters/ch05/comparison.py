"""Comparison of smoothing filters on EEG signals.

Applies a moving average (window=11), Gaussian (sigma=5), and median
(window=11) filter to the P4 channel of the local auditory EEG dataset
and plots all results together as static PNG figures.

Usage:
    python comparison.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
MA_WINDOW = 11
GAUSS_SIGMA = 5
MED_WINDOW = 11


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_plot = 5000
    t_ms = timestamps[:n_plot]

    ma_filtered = moving_average(channel_data, MA_WINDOW)
    gauss_filtered = gaussian_filter1d(channel_data, sigma=GAUSS_SIGMA)
    med_filtered = medfilt(channel_data, kernel_size=MED_WINDOW)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_ms, channel_data[:n_plot], color="gray", linewidth=0.5)
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Original (P4)")

    axes[1].plot(t_ms, ma_filtered[:n_plot], color="blue", linewidth=0.5)
    axes[1].set_ylabel("EEG (uV)")
    axes[1].set_title(f"Moving average (window={MA_WINDOW})")

    axes[2].plot(t_ms, gauss_filtered[:n_plot], color="green", linewidth=0.5)
    axes[2].set_ylabel("EEG (uV)")
    axes[2].set_title(f"Gaussian (sigma={GAUSS_SIGMA})")

    axes[3].plot(t_ms, med_filtered[:n_plot], color="red", linewidth=0.5)
    axes[3].set_ylabel("EEG (uV)")
    axes[3].set_title(f"Median (window={MED_WINDOW})")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Smoothing Filters Comparison - Channel P4", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(__file__).resolve().parent / "comparison_result.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
