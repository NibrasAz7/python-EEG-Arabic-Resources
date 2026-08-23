"""Prepare MOABB dataset for machine learning.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, reshapes from 3D (trials, channels, samples) to 2D
(trials, features), and applies standardization. Visualizes the
distribution before and after scaling.

Usage:
    python prepare_dataset.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials, n_channels * n_samples)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)

    print(f"Original shape: {X.shape}")
    print(f"Reshaped shape: {X_2d.shape}")
    print(f"Scaled shape: {X_scaled.shape}")
    print(f"Classes: {np.unique(labels)}")
    print(f"Trials per class: {[(c, np.sum(labels == c)) for c in np.unique(labels)]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].hist(X_2d[:, :1000].flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Raw feature distribution (first 1000 features)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(X_scaled[:, :1000].flatten(), bins=100, color='orange', alpha=0.7)
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Standardized feature distribution (first 1000 features)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Dataset Preparation - Standardization', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'prepare_dataset_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
