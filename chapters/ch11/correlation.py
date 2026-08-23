"""Compute and plot the channel correlation matrix.

Loads the BNCI2014-001 motor imagery dataset (subject 1), extracts
epochs with the MotorImagery paradigm, computes the average correlation
matrix across all 22 channels (averaged over trials), and plots it as
a heatmap using plt.imshow with a colorbar.

Usage:
    python correlation.py
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

    n_trials, n_channels, n_samples = X.shape

    corr_sum = np.zeros((n_channels, n_channels))
    for trial in range(n_trials):
        trial_data = X[trial]
        std = np.std(trial_data, axis=1, keepdims=True)
        std[std == 0] = 1.0
        normed = (trial_data - np.mean(trial_data, axis=1, keepdims=True)) / std
        corr_sum += np.corrcoef(normed)

    corr_matrix = corr_sum / n_trials

    sessions = ds.get_data(subjects=[1])
    first_run = next(iter(next(iter(next(iter(sessions.values())).values())).values()))
    raw_ch_names = first_run.ch_names
    eog_indices = [i for i, n in enumerate(raw_ch_names) if n.startswith('EOG') or n == 'STI']
    labels_list = [n for i, n in enumerate(raw_ch_names) if i not in eog_indices]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(labels_list, rotation=90, fontsize=7)
    ax.set_yticklabels(labels_list, fontsize=7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    ax.set_title('Channel Correlation Matrix - BNCI2014-001')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson correlation')

    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
