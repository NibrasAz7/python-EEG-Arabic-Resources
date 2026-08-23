"""Empirical Mode Decomposition of EEG signals.

Applies EMD to the P4 channel of the local auditory EEG dataset
using PyEMD, decomposing the signal into Intrinsic Mode Functions
(IMFs) that separate fast artifacts from slower brain activity.

Usage:
    python emd_artifact.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
N_IMFS = 5


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:N_PLOT, 0]

    emd = EMD()
    imfs = emd(channel_data, max_imf=N_IMFS)

    time_sec = np.arange(N_PLOT) / FS
    n_imfs = min(imfs.shape[0], N_IMFS)

    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 12))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='black')
    axes[0].set_ylabel('Original')
    axes[0].set_title('EMD Decomposition - Channel P4', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    for i in range(n_imfs):
        axes[i + 1].plot(time_sec, imfs[i], linewidth=0.5, color='blue')
        axes[i + 1].set_ylabel(f'IMF {i + 1}')
        axes[i + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'emd_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
