"""Train K-Nearest Neighbors classifier on band power features.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, computes band power features (44 features), splits into
train/test, and evaluates KNN with different k values.

Usage:
    python knn.py
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250
BANDS = [(8, 13, 'alpha'), (13, 30, 'beta')]
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]


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

    accuracies = []
    best_k = K_VALUES[0]
    best_acc = 0.0
    best_cm = None

    for k in K_VALUES:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"k={k}: accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_cm = confusion_matrix(y_test, y_pred, labels=['left_hand', 'right_hand'])

    print(f"Best k: {best_k}, accuracy: {best_acc:.4f}")

    classes = ['left_hand', 'right_hand']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(K_VALUES, accuracies, 'o-', color='steelblue', markersize=8)
    axes[0].set_xlabel('k value')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs k Value')
    axes[0].grid(True, alpha=0.3)

    plot_confusion_matrix(axes[1], best_cm, classes)
    axes[1].set_title(f'Confusion Matrix (best k={best_k})')

    plt.suptitle('KNN Classification - Effect of k', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'knn_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

