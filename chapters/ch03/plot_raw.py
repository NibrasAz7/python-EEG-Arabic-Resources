"""Plot raw EEG signal in the time domain (interactive Plotly version).

Plots all 4 channels as separate subplots showing amplitude vs time.
Reveals drift, artifacts, and frequency content visually.

Usage:
    python plot_raw.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    time_sec = np.arange(eeg_data.shape[0]) / FS

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=[f'{ch}' for ch in ch_names],
    )
    for i, ch in enumerate(ch_names):
        fig.add_trace(
            go.Scatter(
                x=time_sec, y=eeg_data[:, i],
                mode='lines', line=dict(width=0.5, color='steelblue'),
                name=ch,
            ),
            row=i + 1, col=1,
        )
        fig.update_yaxes(title_text=f'{ch} (uV)', row=i + 1, col=1)

    fig.update_layout(
        title='Raw EEG Signal - Subject 7, Experiment 1',
        template='plotly',
        showlegend=False,
        height=800,
    )
    fig.update_xaxes(title_text='Time (s)', row=4, col=1)

    fig.show()
    fig.write_html(
        Path(__file__).resolve().parent / 'raw_eeg_plot.html',
        include_plotlyjs='cdn',
    )


if __name__ == "__main__":
    main()
