"""ICA decomposition of multi-channel EEG for artifact separation.

Applies Independent Component Analysis to the 4-channel local
auditory EEG dataset using MNE-Python, separating brain activity
from biological artifacts (EOG, EMG, ECG).

Usage:
    python ica_artifact.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import mne

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    info = mne.create_info(ch_names, sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.T * 1e-6, info, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=4, random_state=97, max_iter=800, verbose=False
    )
    ica.fit(raw, verbose=False)

    components = ica.get_sources(raw).get_data()

    time_sec = np.arange(N_PLOT) / FS
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for i in range(4):
        offset = i * 200
        axes[0].plot(time_sec, eeg_data[:N_PLOT, i] + offset,
                     linewidth=0.5, color=colors[i], label=ch_names[i])
    axes[0].set_yticks([])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Channels')
    axes[0].set_title('Original EEG Signal')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    for i in range(4):
        offset = i * 200
        axes[1].plot(time_sec, components[i, :N_PLOT] * 1e6 + offset,
                     linewidth=0.5, color=colors[i], label=f'IC{i}')
    axes[1].set_yticks([])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Components')
    axes[1].set_title('ICA Independent Components')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('ICA Decomposition - Artifact Separation', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'ica_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
