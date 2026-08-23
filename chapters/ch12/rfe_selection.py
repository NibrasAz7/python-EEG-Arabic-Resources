"""Channel selection by Recursive Feature Elimination (RFE).

Computes band power features (alpha and beta) for each channel of
BNCI2014-001 subject 1, then applies RFE with a linear SVM to select
the 10 most important channel-band features.

Usage:
    python rfe_selection.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250
N_SELECT = 10
BANDS = [(8, 13, 'alpha'), (13, 30, 'beta')]


def compute_band_features(X):
    n_trials, n_channels, n_samples = X.shape
    n_features = n_channels * len(BANDS)
    features = np.zeros((n_trials, n_features))
    for trial in range(n_trials):
        for ch in range(n_channels):
            freqs, psd = welch(X[trial, ch, :], fs=FS, nperseg=256)
            for b_idx, (fmin, fmax, bname) in enumerate(BANDS):
                mask = (freqs >= fmin) & (freqs <= fmax)
                power = np.trapezoid(psd[mask], freqs[mask])
                features[trial, ch * len(BANDS) + b_idx] = power
    return features


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
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    features = compute_band_features(X)
    n_channels = X.shape[1]
    feature_names = []
    for ch in range(n_channels):
        for _, _, bname in BANDS:
            feature_names.append(f'Ch{ch}_{bname}')

    estimator = SVC(kernel='linear')
    selector = RFE(estimator, n_features_to_select=N_SELECT)
    selector.fit(features, labels)
    rankings = selector.ranking_
    selected_mask = selector.support_

    ch_names, pos_2d = get_channel_positions()
    selected_channels = set()
    for i, is_sel in enumerate(selected_mask):
        if is_sel:
            ch_idx = i // len(BANDS)
            selected_channels.add(ch_idx)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sorted_idx = np.argsort(rankings)
    colors = ['green' if selected_mask[i] else 'gray' for i in sorted_idx]
    axes[0].bar(range(len(rankings)), rankings[sorted_idx], color=colors, edgecolor='black')
    axes[0].set_xlabel('Feature (sorted by RFE rank)')
    axes[0].set_ylabel('RFE rank (1 = best)')
    axes[0].set_title('RFE feature ranking')
    axes[0].grid(True, alpha=0.3, axis='y')

    head = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    axes[1].add_patch(head)
    nose = plt.Polygon([[0, 1.0], [-0.08, 1.12], [0.08, 1.12]], fill=False, color='black', linewidth=1.5)
    axes[1].add_patch(nose)
    for i in range(n_channels):
        color = 'green' if i in selected_channels else 'red'
        axes[1].scatter(pos_2d[i, 0], pos_2d[i, 1], c=color, s=100, zorder=5, edgecolors='black')
        axes[1].annotate(ch_names[i], (pos_2d[i, 0], pos_2d[i, 1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Selected (green) vs rejected (red) channels')

    plt.suptitle('Channel Selection by RFE (Recursive Feature Elimination)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'rfe_selection_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
