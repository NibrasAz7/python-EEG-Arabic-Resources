"""Median filter for EEG signals with artificial spike.

Applies a median filter with three window sizes (5, 11, 21) to the
P4 channel of the local auditory EEG dataset after injecting an
artificial spike at sample 1000 to demonstrate outlier resistance.
Results are plotted as static PNG figures.

Usage:
    python median_filter.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
WINDOWS = [5, 11, 21]
SPIKE_INDEX = 1000
SPIKE_VALUE = 200.0


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0].copy()
    channel_data[SPIKE_INDEX] += SPIKE_VALUE

    n_plot = 5000
    t_ms = timestamps[:n_plot]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_ms, channel_data[:n_plot], color="gray", linewidth=0.5)
    axes[0].plot(t_ms[SPIKE_INDEX], channel_data[SPIKE_INDEX], "ro", markersize=4)
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Original with spike (P4)")

    for ax, w in zip(axes[1:], WINDOWS):
        filtered = medfilt(channel_data, kernel_size=w)
        ax.plot(t_ms, filtered[:n_plot], color="blue", linewidth=0.5)
        ax.set_ylabel("EEG (uV)")
        ax.set_title(f"Median (window={w})")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Median Filter - Channel P4 with Artificial Spike", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(__file__).resolve().parent / "median_filter_result.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
