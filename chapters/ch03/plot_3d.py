"""3D visualization of EEG data.

Plots signal amplitude as a 3D scatter plot with time, channel, and
amplitude axes. Subsamples data for visual clarity.

Usage:
    python plot_3d.py
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

    # Subsample for clarity (every 10th sample)
    step = 10
    time_sec = np.arange(0, eeg_data.shape[0], step) / FS
    data_sub = eeg_data[::step, :]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, ch in enumerate(ch_names):
        ax.scatter(
            time_sec,
            np.full_like(time_sec, i),
            data_sub[:, i],
            c=data_sub[:, i],
            cmap='viridis',
            s=1,
            label=ch
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.set_zlabel('Amplitude (uV)')
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names)
    ax.set_title('3D EEG Visualization - Subject 7')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'eeg_3d.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
