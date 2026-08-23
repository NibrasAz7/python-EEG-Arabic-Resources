"""Entropy analysis of EEG signals using sliding windows.

Computes sample entropy, approximate entropy, and spectral entropy
on sliding windows of the P4 channel to measure signal complexity
variations over time. Uses the antropy library.

Usage:
    python entropy_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from antropy import sample_entropy, app_entropy, spectral_entropy

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
WINDOW = 1000
STEP = 500
M = 2
R = 0.2


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_windows = (len(channel_data) - WINDOW) // STEP + 1
    sampen_vals = []
    appen_vals = []
    specen_vals = []
    window_centers = []

    for i in range(n_windows):
        start = i * STEP
        end = start + WINDOW
        segment = np.ascontiguousarray(channel_data[start:end])
        r_val = R * np.std(segment)

        sampen = sample_entropy(segment, order=M, tolerance=r_val)
        appen = app_entropy(segment, order=M, tolerance=r_val)
        specen = spectral_entropy(segment, sf=FS, method='welch', normalize=True)

        sampen_vals.append(sampen)
        appen_vals.append(appen)
        specen_vals.append(specen)
        window_centers.append((start + end) / 2 / FS)

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(time_sec, channel_data[:N_PLOT], linewidth=0.5, color='blue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(window_centers, sampen_vals, linewidth=1.5, color='green', label='Sample entropy')
    axes[1].plot(window_centers, appen_vals, linewidth=1.5, color='orange', label='Approximate entropy')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Sample and Approximate Entropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(window_centers, specen_vals, linewidth=1.5, color='purple')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Spectral entropy')
    axes[2].set_title('Spectral Entropy (normalized)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Entropy Analysis - Channel P4', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'entropy_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
