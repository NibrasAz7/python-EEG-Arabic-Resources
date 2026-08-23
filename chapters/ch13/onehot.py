"""Demonstrate one-hot encoding for multi-class EEG labels.

Loads BNCI2014-001 subject 1 with all 4 motor imagery classes, applies
LabelEncoder and OneHotEncoder, and visualizes class distribution and
the one-hot encoded matrix.

Usage:
    python onehot.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    print(f"X shape: {X.shape}")
    print(f"Original labels: {np.unique(labels)}")
    print(f"Class distribution: {[(c, np.sum(labels == c)) for c in np.unique(labels)]}")

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_matrix = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))

    print(f"Encoded labels: {np.unique(labels_encoded)}")
    print(f"One-hot matrix shape: {onehot_matrix.shape}")

    classes = label_encoder.classes_

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    counts = [np.sum(labels == c) for c in classes]
    axes[0].bar(range(len(classes)), counts, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution (4 classes)')
    axes[0].grid(True, alpha=0.3, axis='y')

    im = axes[1].imshow(onehot_matrix[:20, :], cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].set_yticks(range(20))
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Trial')
    axes[1].set_title('One-Hot Encoded Matrix (first 20 trials)')
    plt.colorbar(im, ax=axes[1])

    plt.suptitle('One-Hot Encoding of EEG Labels', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'onehot_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

