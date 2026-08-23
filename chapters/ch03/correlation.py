"""Compute and plot the correlation matrix between EEG channels (interactive Plotly version).

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
import plotly.graph_objects as go
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    # Compute correlation matrix
    corr_matrix = np.corrcoef(eeg_data.T)

    # Build text annotations for each cell
    text = [[f'{corr_matrix[i, j]:.2f}' for j in range(len(ch_names))]
            for i in range(len(ch_names))]

    # Plot
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=ch_names,
        y=ch_names,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title='Correlation'),
    ))
    fig.update_layout(
        title='EEG Channel Correlation Matrix',
        xaxis_title='Channel',
        yaxis_title='Channel',
        template='plotly',
    )
    fig.update_yaxes(autorange='reversed')

    fig.show()
    fig.write_html(
        Path(__file__).resolve().parent / 'correlation_matrix.html',
        include_plotlyjs='cdn',
    )

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
