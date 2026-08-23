"""Plot EEG frequency bands using Welch's method (interactive Plotly version).

Computes the power spectral density of the P4 channel from the local
auditory EEG dataset and highlights the five main frequency bands
(delta, theta, alpha, beta, gamma). Uses Plotly for interactive hover.

Usage:
    python plot_frequency_bands.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import signal

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)

FREQ_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4, "purple"),
    "Theta (4-8 Hz)":   (4, 8, "blue"),
    "Alpha (8-13 Hz)":  (8, 13, "green"),
    "Beta (13-30 Hz)":  (13, 30, "orange"),
    "Gamma (30-100 Hz)": (30, 100, "red"),
}


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]  # P4

    freqs, psd = signal.welch(channel_data, fs=FS, nperseg=1024)

    fig = go.Figure()

    # Main PSD curve (log scale y-axis)
    fig.add_trace(go.Scatter(
        x=freqs, y=psd, mode="lines", name="PSD",
        line=dict(color="black", width=1),
        hovertemplate="Freq: %{x:.1f} Hz<br>Power: %{y:.2e}<extra></extra>",
    ))

    # Band shading
    for name, (lo, hi, color) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        fig.add_trace(go.Scatter(
            x=freqs[mask], y=psd[mask],
            fill="tozeroy", mode="none",
            name=name, fillcolor=color,
            opacity=0.3,
            hovertemplate=f"{name}<br>Freq: %{{x:.1f}} Hz<br>Power: %{{y:.2e}}<extra></extra>",
        ))

    fig.update_layout(
        title="EEG Frequency Bands (P4 channel, subject 7)",
        xaxis_title="Frequency (Hz)", yaxis_title="Power (uV^2 / Hz)",
        xaxis=dict(range=[0, 100]), yaxis_type="log",
        height=500, hovermode="x unified",
    )

    out = Path(__file__).resolve().parent / "freq_bands_result.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved: {out}")

    fig.show()


if __name__ == "__main__":
    main()
