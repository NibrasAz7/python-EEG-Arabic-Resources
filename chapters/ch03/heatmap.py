"""Create a heatmap of EEG data.

Displays signal amplitude across channels and time using a color-coded
2D grid. Reveals patterns that are hard to see in line plots.

Usage:
    python heatmap.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        eeg_data.T,  # shape: (channels, samples)
        aspect='auto',
        cmap='RdBu_r',
        vmin=-100, vmax=100,
        extent=[0, eeg_data.shape[0] / FS, 4, 0]
    )
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(ch_names)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.set_title('EEG Heatmap - Subject 7')
    plt.colorbar(im, ax=ax, label='Amplitude (uV)')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'eeg_heatmap.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
