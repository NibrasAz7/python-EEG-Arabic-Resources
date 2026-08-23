"""Gaussian filter for EEG signals.

Applies a 1D Gaussian filter with three sigma values (2, 5, 10)
to the P4 channel of the local auditory EEG dataset and plots the
results as static PNG figures.

Usage:
    python gaussian_filter.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
SIGMAS = [2, 5, 10]


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_plot = 5000
    t_ms = timestamps[:n_plot]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_ms, channel_data[:n_plot], color="gray", linewidth=0.5)
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Original (P4)")

    for ax, s in zip(axes[1:], SIGMAS):
        filtered = gaussian_filter1d(channel_data, sigma=s)
        ax.plot(t_ms, filtered[:n_plot], color="blue", linewidth=0.5)
        ax.set_ylabel("EEG (uV)")
        ax.set_title(f"Gaussian (sigma={s})")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Gaussian Filter - Channel P4", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(__file__).resolve().parent / "gaussian_filter_result.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
