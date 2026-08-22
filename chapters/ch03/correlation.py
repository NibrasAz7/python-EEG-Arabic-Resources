"""Compute and plot the correlation matrix between EEG channels.

Calculates Pearson correlation coefficients between all pairs of
channels and displays them as a color-coded matrix with annotated
values.

Usage:
    python correlation.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    # Compute correlation matrix
    corr_matrix = np.corrcoef(eeg_data.T)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(ch_names)))
    ax.set_yticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names)
    ax.set_yticklabels(ch_names)
    ax.set_title('EEG Channel Correlation Matrix')

    # Annotate each cell with the correlation value
    for i in range(len(ch_names)):
        for j in range(len(ch_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=10)

    plt.colorbar(im, label='Correlation')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'correlation_matrix.png', dpi=150)
    plt.show()

    # Print numerical values
    print("Correlation matrix:")
    print(f"{'':>6}", end="")
    for ch in ch_names:
        print(f"{ch:>8}", end="")
    print()
    for i, ch in enumerate(ch_names):
        print(f"{ch:>6}", end="")
        for j in range(len(ch_names)):
            print(f"{corr_matrix[i, j]:>8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
