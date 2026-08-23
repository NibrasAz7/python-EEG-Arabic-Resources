"""Hyperparameter tuning with GridSearchCV for Logistic Regression.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, computes band power features (44 features), and performs
grid search over C, penalty, and solver parameters with 5-fold
cross-validation.

Usage:
    python grid_search.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
    }

    clf = LogisticRegression(max_iter=1000, random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid.fit(features_scaled, labels)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    results = grid.cv_results_
    C_values = [0.01, 0.1, 1, 10, 100]
    penalties = ['l1', 'l2']

    mean_acc_per_C = np.zeros(len(C_values))
    for i, c in enumerate(C_values):
        mask_c = results['param_C'] == c
        mean_acc_per_C[i] = np.mean(results['mean_test_score'][mask_c])

    pivot = np.zeros((len(penalties), len(C_values)))
    for i, p in enumerate(penalties):
        for j, c in enumerate(C_values):
            mask_p = (results['param_penalty'] == p) & (results['param_C'] == c)
            pivot[i, j] = results['mean_test_score'][mask_p][0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].bar(range(len(C_values)), mean_acc_per_C, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(C_values)))
    axes[0].set_xticklabels([str(c) for c in C_values])
    axes[0].set_xlabel('C value')
    axes[0].set_ylabel('Mean CV Accuracy')
    axes[0].set_title('Mean CV Accuracy per C (averaged across penalties)')
    axes[0].grid(True, alpha=0.3, axis='y')

    im = axes[1].imshow(pivot, cmap='YlGnBu', aspect='auto')
    axes[1].set_xticks(range(len(C_values)))
    axes[1].set_xticklabels([str(c) for c in C_values])
    axes[1].set_yticks(range(len(penalties)))
    axes[1].set_yticklabels(penalties)
    axes[1].set_xlabel('C')
    axes[1].set_ylabel('Penalty')
    axes[1].set_title('CV Results (C x Penalty)')
    for i in range(len(penalties)):
        for j in range(len(C_values)):
            axes[1].text(j, i, f'{pivot[i, j]:.3f}', ha='center', va='center', fontsize=11)
    plt.colorbar(im, ax=axes[1])

    plt.suptitle('Grid Search Hyperparameter Tuning', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'grid_search_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

