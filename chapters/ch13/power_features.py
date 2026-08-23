"""Compute band power features for machine learning.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, computes alpha and beta band power features for 22 channels
(44 features total), and visualizes the feature matrix and mean power
per channel.

Usage:
    python power_features.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250
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


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    features = compute_band_features(X)
    print(f"Feature matrix shape: {features.shape}")

    n_channels = X.shape[1]
    mean_power = np.zeros(n_channels)
    for ch in range(n_channels):
        alpha_idx = ch * len(BANDS) + 0
        beta_idx = ch * len(BANDS) + 1
        mean_power[ch] = np.mean(features[:, alpha_idx]) + np.mean(features[:, beta_idx])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    im = axes[0].imshow(features[:50, :], aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Feature (channel x band)')
    axes[0].set_ylabel('Trial')
    axes[0].set_title('Feature Matrix (first 50 trials x 44 features)')
    plt.colorbar(im, ax=axes[0])

    axes[1].bar(range(n_channels), mean_power, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Mean Power (alpha + beta)')
    axes[1].set_title('Mean Power per Channel')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Band Power Features for ML', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'power_features_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

