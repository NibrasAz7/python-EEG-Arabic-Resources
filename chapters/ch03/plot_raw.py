"""Plot raw EEG signal in the time domain.

Plots all 4 channels as separate subplots showing amplitude vs time.
Reveals drift, artifacts, and frequency content visually.

Usage:
    python plot_raw.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    time_sec = np.arange(eeg_data.shape[0]) / FS

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for i, ch in enumerate(ch_names):
        axes[i].plot(time_sec, eeg_data[:, i], linewidth=0.5, color='steelblue')
        axes[i].set_ylabel(f'{ch} (uV)')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Raw EEG Signal - Subject 7, Experiment 1')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'raw_eeg_plot.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
