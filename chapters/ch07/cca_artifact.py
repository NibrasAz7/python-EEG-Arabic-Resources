"""Regression-based removal of powerline interference from EEG.

Builds a reference matrix of 50 Hz sine/cosine (and 2nd harmonic)
signals, uses least-squares regression to estimate the artifact
component in each EEG channel, and subtracts it to remove powerline
interference from the local auditory EEG dataset.

Usage:
    python cca_artifact.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
POWERLINE_FREQ = 50.0


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_samples = len(channel_data)
    t = np.arange(n_samples) / FS
    # Reference: 50 Hz fundamental + 2nd harmonic (sine and cosine)
    reference = np.column_stack([
        np.sin(2 * np.pi * POWERLINE_FREQ * t),
        np.cos(2 * np.pi * POWERLINE_FREQ * t),
        np.sin(2 * np.pi * 2 * POWERLINE_FREQ * t),
        np.cos(2 * np.pi * 2 * POWERLINE_FREQ * t),
    ])

    # Regression-based subtraction: fit how much of the EEG channel
    # is explained by the powerline reference, then subtract it.
    reg = LinearRegression()
    reg.fit(reference, channel_data)
    artifact = reg.predict(reference)
    cleaned = channel_data - artifact

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data[:N_PLOT], linewidth=0.5, color='blue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_sec, cleaned[:N_PLOT], linewidth=0.5, color='green', label='Cleaned')
    axes[1].plot(time_sec, artifact[:N_PLOT], linewidth=1, color='red', label='Removed component')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].set_title('Regression-based Cleaned Signal (50 Hz removed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Regression-based Artifact Removal - Powerline Interference (50 Hz)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'cca_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
