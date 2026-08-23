"""Plot raw EEG trials from MOABB dataset.

Loads BNCI2014-001 subject 1, extracts motor imagery epochs,
and plots one trial from each class (left hand, right hand)
showing all 22 channels with vertical offsets.

Usage:
    python plot_dataset.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250
N_PLOT_SAMPLES = 1000


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    left_idx = np.where(labels == 'left_hand')[0][0]
    right_idx = np.where(labels == 'right_hand')[0][0]

    left_trial = X[left_idx, :, :N_PLOT_SAMPLES]
    right_trial = X[right_idx, :, :N_PLOT_SAMPLES]

    n_channels = left_trial.shape[0]
    time_sec = np.arange(N_PLOT_SAMPLES) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ch in range(n_channels):
        offset = ch * 50
        axes[0].plot(time_sec, left_trial[ch, :] + offset, linewidth=0.5)
    axes[0].set_yticks([])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Channels')
    axes[0].set_title('Motor Imagery - Left Hand')
    axes[0].grid(True, alpha=0.3)

    for ch in range(n_channels):
        offset = ch * 50
        axes[1].plot(time_sec, right_trial[ch, :] + offset, linewidth=0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Channels')
    axes[1].set_title('Motor Imagery - Right Hand')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('MOABB BNCI2014-001 - Motor Imagery Trials', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'plot_dataset_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
