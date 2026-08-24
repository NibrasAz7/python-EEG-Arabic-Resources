"""Prepare MOABB dataset for machine learning.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, applies a 50 Hz notch filter to remove powerline noise,
reshapes from 3D (trials, channels, samples) to 2D
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
from scipy.signal import iirnotch, filtfilt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250


def apply_notch_filter(X, freq=50.0, fs=FS, Q=30.0):
    b, a = iirnotch(freq, Q, fs=fs)
    n_trials, n_channels, n_samples = X.shape
    X_filtered = np.zeros_like(X)
    for trial in range(n_trials):
        for ch in range(n_channels):
            X_filtered[trial, ch, :] = filtfilt(b, a, X[trial, ch, :])
    return X_filtered


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    X = apply_notch_filter(X, freq=50.0)

    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials, n_channels * n_samples)

    # NOTE: scaler is fit on training data only to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, labels, test_size=0.2, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Original shape: {X.shape}")
    print(f"Reshaped shape: {X_2d.shape}")
    print(f"Train scaled shape: {X_train_scaled.shape}")
    print(f"Test scaled shape: {X_test_scaled.shape}")
    print(f"Classes: {np.unique(labels)}")
    print(f"Trials per class: {[(c, np.sum(labels == c)) for c in np.unique(labels)]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].hist(X_2d[:, :1000].flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Filtered feature distribution (first 1000 features)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(X_train_scaled[:, :1000].flatten(), bins=100, color='orange', alpha=0.7)
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Standardized feature distribution - train (first 1000 features)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Dataset Preparation - Notch Filter + Standardization', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'prepare_dataset_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
