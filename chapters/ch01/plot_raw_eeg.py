"""Plot raw EEG signal from all four channels.

Loads subject 7 from the local auditory EEG dataset and plots
the raw signal from all four channels (P4, Cz, F8, T7) in a
stacked subplot layout.

Usage:
    python plot_raw_eeg.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)
N_PLOT = 5000  # Number of samples to plot (~25 seconds at 200 Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    n_plot = min(N_PLOT, eeg_data.shape[0])
    t_sec = timestamps[:n_plot] / 1000.0

    fig, axes = plt.subplots(
        eeg_data.shape[1], 1, figsize=(14, 10), sharex=True
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (ax, ch, color) in enumerate(zip(axes, ch_names, colors)):
        ax.plot(t_sec, eeg_data[:n_plot, i], color=color, linewidth=0.5)
        ax.set_ylabel(f"{ch} (uV)")
        ax.set_title(f"Channel {ch}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Raw EEG Signal - Subject 7 ({n_plot} samples, "
        f"{n_plot / FS:.1f} s)",
        fontsize=14,
    )
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "raw_eeg_result.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
