"""Channel selection by Pearson correlation filtering.

Computes the average correlation matrix across trials of BNCI2014-001
subject 1, identifies pairs of channels with |r| > 0.85, and removes
the channel with lower variance from each pair.

Usage:
    python pearson_selection.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent
THRESHOLD = 0.85


def get_channel_positions():
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[1])
    subject_data = data[1]
    session_key = list(subject_data.keys())[0]
    run_key = list(subject_data[session_key].keys())[0]
    raw = subject_data[session_key][run_key]
    montage = raw.get_montage()
    ch_pos = montage.get_positions()['ch_pos']
    ch_names = [ch for ch in raw.ch_names if ch in ch_pos and not ch.startswith('EOG')]
    positions = np.array([ch_pos[ch] for ch in ch_names])
    pos_2d = positions[:, :2]
    scale = 1.0 / np.max(np.abs(pos_2d))
    pos_2d = pos_2d * scale * 0.95
    return ch_names, pos_2d


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    n_trials, n_channels, n_samples = X.shape

    corr_sum = np.zeros((n_channels, n_channels))
    for trial in range(n_trials):
        corr = np.corrcoef(X[trial, :, :])
        corr_sum += corr
    corr_avg = corr_sum / n_trials

    variances = np.var(X.reshape(n_trials, n_channels, n_samples), axis=(0, 2))
    to_remove = set()
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if abs(corr_avg[i, j]) > THRESHOLD:
                if variances[i] < variances[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    kept = [i for i in range(n_channels) if i not in to_remove]

    ch_names, pos_2d = get_channel_positions()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im = axes[0].imshow(corr_avg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in to_remove:
        axes[0].plot(i, i, 'rx', markersize=15, markeredgewidth=3)
    axes[0].set_xlabel('Channel index')
    axes[0].set_ylabel('Channel index')
    axes[0].set_title('Correlation matrix (red X = removed)')
    plt.colorbar(im, ax=axes[0], label='Correlation')

    head = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    axes[1].add_patch(head)
    nose = plt.Polygon([[0, 1.0], [-0.08, 1.12], [0.08, 1.12]], fill=False, color='black', linewidth=1.5)
    axes[1].add_patch(nose)
    for i in range(n_channels):
        color = 'green' if i in kept else 'red'
        axes[1].scatter(pos_2d[i, 0], pos_2d[i, 1], c=color, s=100, zorder=5, edgecolors='black')
        axes[1].annotate(ch_names[i], (pos_2d[i, 0], pos_2d[i, 1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Kept (green) vs removed (red) - {len(kept)} channels')

    plt.suptitle('Channel Selection by Pearson Correlation', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'pearson_selection_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
