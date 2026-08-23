"""Compare standardization (Z-score) vs normalization (Min-Max).

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, reshapes to 2D, applies StandardScaler and MinMaxScaler,
and visualizes the distributions before and after scaling.

Usage:
    python standardize.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler, MinMaxScaler

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    n_trials, n_channels, n_samples = X.shape
    X_2d = X.reshape(n_trials, n_channels * n_samples)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_2d)

    normalizer = MinMaxScaler()
    X_norm = normalizer.fit_transform(X_2d)

    print(f"Original shape: {X_2d.shape}")
    print(f"Raw mean: {X_2d.mean():.4f}, std: {X_2d.std():.4f}")
    print(f"Standardized mean: {X_std.mean():.4f}, std: {X_std.std():.4f}")
    print(f"Normalized min: {X_norm.min():.4f}, max: {X_norm.max():.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    axes[0].hist(X_2d[:, :1000].flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Raw feature distribution (first 1000 features)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(X_std[:, :1000].flatten(), bins=100, color='orange', alpha=0.7)
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Standardized distribution (Z-score)')
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(X_norm[:, :1000].flatten(), bins=100, color='green', alpha=0.7)
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Normalized distribution (Min-Max, 0-1)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Standardization vs Normalization', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'standardize_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

