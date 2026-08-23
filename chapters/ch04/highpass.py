"""High-pass Butterworth filter for EEG signals (interactive Plotly version).

Applies a 1 Hz high-pass filter to the P4 channel of the local
auditory EEG dataset and plots the result before and after filtering.

Usage:
    python highpass.py
"""

from pathlib import Path

import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)
CUTOFF = 1.0  # High-pass cutoff (Hz)
ORDER = 4  # Butterworth filter order


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a high-pass Butterworth filter using filtfilt (zero-phase).

    Args:
        data: 1D EEG signal array.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]  # P4

    filtered = butter_highpass_filter(channel_data, cutoff=CUTOFF, fs=FS, order=ORDER)

    n_plot = 5000
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Before high-pass filter", f"After high-pass filter ({CUTOFF} Hz)"),
    )
    fig.add_trace(
        go.Scatter(x=timestamps[:n_plot], y=channel_data[:n_plot], name="Raw"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps[:n_plot], y=filtered[:n_plot],
            name="Filtered", line=dict(color="green"),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="EEG (uV)", row=1, col=1)
    fig.update_yaxes(title_text="EEG (uV)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_layout(template="plotly", showlegend=False)

    fig.show()
    out = Path(__file__).resolve().parent / "highpass_result.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
