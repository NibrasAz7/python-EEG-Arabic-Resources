"""Train Random Forest classifier on band power features.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, computes band power features (44 features), splits into
train/test, trains RandomForestClassifier, and visualizes the
confusion matrix and top feature importances.

Usage:
    python random_forest.py
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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


def plot_confusion_matrix(ax, cm, classes):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    plt.colorbar(im, ax=ax)


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    features = compute_band_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['left_hand', 'right_hand'])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion matrix:\n{cm}")

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_20 = importances[sorted_idx[:20]]
    classes = ['left_hand', 'right_hand']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    plot_confusion_matrix(axes[0], cm, classes)
    axes[0].set_title('Confusion Matrix')

    colors = ['green' if i < 10 else 'steelblue' for i in range(20)]
    axes[1].bar(range(20), top_20, color=colors, edgecolor='black')
    axes[1].set_xlabel('Feature (sorted by importance)')
    axes[1].set_ylabel('Importance')
    axes[1].set_title('Top 20 Feature Importances (top 10 in green)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Random Forest Classification', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'random_forest_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

