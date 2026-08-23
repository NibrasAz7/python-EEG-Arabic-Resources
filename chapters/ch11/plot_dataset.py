"""Plot raw EEG trials from a MOABB dataset.

Loads the BNCI2014-001 motor imagery dataset (subject 1), extracts
epochs with the MotorImagery paradigm, and plots two subplots: one
trial of left-hand motor imagery and one trial of right-hand motor
imagery, showing the first 5 seconds across all 22 channels with
vertical offsets.

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


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    ds = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=ds, subjects=[1])

    sfreq = 250
    n_samples = X.shape[2]
    epoch_dur = n_samples / sfreq
    plot_samples = min(int(5 * sfreq), n_samples)
    time = np.arange(plot_samples) / sfreq

    left_idx = np.where(labels == 'left_hand')[0][0]
    right_idx = np.where(labels == 'right_hand')[0][0]

    left_trial = X[left_idx, :, :plot_samples]
    right_trial = X[right_idx, :, :plot_samples]

    n_channels = X.shape[1]
    offset_step = 1.2 * np.max(np.abs(X))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ch in range(n_channels):
        axes[0].plot(time, left_trial[ch] + ch * offset_step, linewidth=0.6)
    axes[0].set_xlim(0, plot_samples / sfreq)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Channels (offset)')
    axes[0].set_title(f'Left hand motor imagery (trial {left_idx}, epoch {epoch_dur:.1f}s)')
    axes[0].grid(True, alpha=0.3)

    for ch in range(n_channels):
        axes[1].plot(time, right_trial[ch] + ch * offset_step, linewidth=0.6)
    axes[1].set_xlim(0, plot_samples / sfreq)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Channels (offset)')
    axes[1].set_title(f'Right hand motor imagery (trial {right_idx}, epoch {epoch_dur:.1f}s)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('MOABB BNCI2014-001 - Motor Imagery Trials', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_dir / 'plot_dataset_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
