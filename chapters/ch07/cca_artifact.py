"""CCA-based removal of powerline interference from EEG.

Uses Canonical Correlation Analysis to find and remove the 50 Hz
powerline component from the P4 channel of the local auditory
EEG dataset by correlating with synthetic sine/cosine references.

Usage:
    python cca_artifact.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

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

    t = np.arange(len(channel_data)) / FS
    ref_sin = np.sin(2 * np.pi * POWERLINE_FREQ * t).reshape(-1, 1)
    ref_cos = np.cos(2 * np.pi * POWERLINE_FREQ * t).reshape(-1, 1)
    reference = np.hstack([ref_sin, ref_cos])

    eeg_2d = channel_data.reshape(-1, 1)
    cca = CCA(n_components=1)
    cca.fit(eeg_2d, reference)
    eeg_c, ref_c = cca.transform(eeg_2d, reference)

    artifact = eeg_c[:, 0]
    artifact = artifact / artifact.std() * channel_data.std()
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
    axes[1].set_title('CCA Cleaned Signal (50 Hz removed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('CCA Artifact Removal - Powerline Interference (50 Hz)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'cca_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
