"""Channel correlation matrix for MOABB dataset.

Computes the average correlation matrix across all trials of
BNCI2014-001 subject 1 and visualizes it as a heatmap.

Usage:
    python correlation.py
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
    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    n_trials, n_channels, n_samples = X.shape

    corr_sum = np.zeros((n_channels, n_channels))
    for trial in range(n_trials):
        trial_data = X[trial, :, :]
        corr = np.corrcoef(trial_data)
        corr_sum += corr
    corr_avg = corr_sum / n_trials

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr_avg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Channel index')
    ax.set_title('Channel Correlation Matrix - BNCI2014-001')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
