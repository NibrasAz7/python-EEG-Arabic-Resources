"""Plot raw EEG signal from all four channels (interactive Plotly version).

Loads subject 7 from the local auditory EEG dataset and plots
the raw signal from all four channels (P4, Cz, F8, T7) in a
stacked subplot layout. Uses Plotly for interactive zoom/pan.

Usage:
    python plot_raw_eeg.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)
N_PLOT = 5000  # Number of samples to plot (~25 seconds at 200 Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    n_plot = min(N_PLOT, eeg_data.shape[0])
    t_sec = timestamps[:n_plot] / 1000.0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = make_subplots(
        rows=eeg_data.shape[1], cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"Channel {ch}" for ch in ch_names],
    )

    for i, (ch, color) in enumerate(zip(ch_names, colors)):
        fig.add_trace(
            go.Scatter(
                x=t_sec, y=eeg_data[:n_plot, i],
                mode="lines", name=f"{ch} (uV)",
                line=dict(color=color, width=0.5),
            ),
            row=i + 1, col=1,
        )
        fig.update_yaxes(title_text=f"{ch} (uV)", row=i + 1, col=1)

    fig.update_layout(
        title_text=f"Raw EEG Signal - Subject 7 ({n_plot} samples, "
                   f"{n_plot / FS:.1f} s)",
        height=800, showlegend=False,
        xaxis_title="Time (s)",
    )
    fig.update_xaxes(title_text="Time (s)", row=eeg_data.shape[1], col=1)

    out = Path(__file__).resolve().parent / "raw_eeg_result.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved: {out}")

    fig.show()


if __name__ == "__main__":
    main()
