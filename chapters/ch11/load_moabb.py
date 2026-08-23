"""Load a MOABB dataset and explore its structure.

Downloads BNCI2014-001 (Motor Imagery) for subject 1, extracts
epochs using the MotorImagery paradigm, and prints the structure.
Saves a bar chart of trial counts per class.

Usage:
    python load_moabb.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    n_trials, n_channels, n_samples = X.shape
    unique_labels = np.unique(labels)
    trials_per_class = {label: np.sum(labels == label) for label in unique_labels}

    info_lines = [
        f"Dataset: BNCI2014-001 (Motor Imagery)",
        f"Subject: 1",
        f"Trials: {n_trials}",
        f"Channels: {n_channels}",
        f"Samples per trial: {n_samples}",
        f"Sampling rate: 250 Hz",
        f"Trial duration: {n_samples / 250:.2f} s",
        f"Classes: {list(unique_labels)}",
        f"Trials per class: {trials_per_class}",
    ]
    info_text = "\n".join(info_lines)
    print(info_text)

    with open(OUTPUT_DIR / "moabb_info.txt", "w") as f:
        f.write(info_text)

    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(trials_per_class.keys())
    counts = list(trials_per_class.values())
    ax.bar(classes, counts, color=['steelblue', 'orange'], edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of trials')
    ax.set_title('Trial count per class - BNCI2014-001 Subject 1')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'load_moabb_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
