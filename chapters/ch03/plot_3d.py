"""3D visualization of EEG data (interactive Plotly version).

Plots signal amplitude as a 3D scatter plot with time, channel, and
amplitude axes. Subsamples data for visual clarity.

Usage:
    python plot_3d.py
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

    # Subsample for clarity (every 10th sample)
    step = 10
    time_sec = np.arange(0, eeg_data.shape[0], step) / FS
    data_sub = eeg_data[::step, :]

    fig = go.Figure()
    for i, ch in enumerate(ch_names):
        fig.add_trace(go.Scatter3d(
            x=time_sec,
            y=np.full_like(time_sec, i),
            z=data_sub[:, i],
            mode='markers',
            marker=dict(
                size=1,
                color=data_sub[:, i],
                colorscale='viridis',
            ),
            name=ch,
        ))

    fig.update_layout(
        title='3D EEG Visualization - Subject 7',
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Channel',
            zaxis_title='Amplitude (uV)',
            yaxis=dict(
                tickvals=list(range(len(ch_names))),
                ticktext=ch_names,
            ),
        ),
        template='plotly',
    )

    fig.show()
    fig.write_html(
        Path(__file__).resolve().parent / 'eeg_3d.html',
        include_plotlyjs='cdn',
    )


if __name__ == "__main__":
    main()
