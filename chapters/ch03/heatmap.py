"""Create a heatmap of EEG data (interactive Plotly version).

Displays signal amplitude across channels and time using a color-coded
2D grid. Reveals patterns that are hard to see in line plots.

Usage:
    python heatmap.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    time_sec = np.arange(eeg_data.shape[0]) / FS

    fig = go.Figure(data=go.Heatmap(
        z=eeg_data.T,
        x=time_sec,
        y=ch_names,
        colorscale='RdBu_r',
        zmin=-100, zmax=100,
        colorbar=dict(title='Amplitude (uV)'),
    ))
    fig.update_layout(
        title='EEG Heatmap - Subject 7',
        xaxis_title='Time (s)',
        yaxis_title='Channel',
        template='plotly',
    )
    fig.update_yaxes(autorange='reversed')

    fig.show()
    fig.write_html(
        Path(__file__).resolve().parent / 'eeg_heatmap.html',
        include_plotlyjs='cdn',
    )


if __name__ == "__main__":
    main()
