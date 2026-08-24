"""Channel selection by Random Forest feature importance.

Computes band power features (alpha and beta) for each channel of
BNCI2014-001 subject 1, trains a RandomForestClassifier, and extracts
feature importances to identify the most important channels.

Usage:
    python feature_importance.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")

    channel_importance = np.zeros(n_channels)
    for ch in range(n_channels):
        band_importances = [importances[ch * len(BANDS) + b] for b in range(len(BANDS))]
        channel_importance[ch] = max(band_importances)

    sorted_idx = np.argsort(channel_importance)[::-1]
    selected = set(sorted_idx[:N_SELECT])

    ch_names, pos_2d = get_channel_positions()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sorted_feat = np.argsort(importances)[::-1]
    colors = ['green' if i < N_SELECT else 'gray' for i in range(len(importances))]
    axes[0].bar(range(len(importances)), importances[sorted_feat], color=colors, edgecolor='black')
    axes[0].set_xlabel('Feature (sorted by importance)')
    axes[0].set_ylabel('Importance')
    axes[0].set_title('Random Forest feature importance')
    axes[0].grid(True, alpha=0.3, axis='y')

    head = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    axes[1].add_patch(head)
    nose = plt.Polygon([[0, 1.0], [-0.08, 1.12], [0.08, 1.12]], fill=False, color='black', linewidth=1.5)
    axes[1].add_patch(nose)
    for i in range(n_channels):
        color = 'green' if i in selected else 'gray'
        size = 50 + channel_importance[i] * 5000
        axes[1].scatter(pos_2d[i, 0], pos_2d[i, 1], c=color, s=size, zorder=5, edgecolors='black')
        axes[1].annotate(ch_names[i], (pos_2d[i, 0], pos_2d[i, 1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Channel importance (size = importance)')

    plt.suptitle('Channel Selection by Random Forest Feature Importance', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'feature_importance_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
