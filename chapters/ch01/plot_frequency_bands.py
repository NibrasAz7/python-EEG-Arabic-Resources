"""Plot EEG frequency bands using Welch's method.

Computes the power spectral density of the P4 channel from the local
auditory EEG dataset and highlights the five main frequency bands
(delta, theta, alpha, beta, gamma).

Usage:
    python plot_frequency_bands.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    # Compute power spectral density using Welch's method
    freqs, psd = signal.welch(channel_data, fs=FS, nperseg=1024)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs, psd, color="black", linewidth=0.8)

    for name, (lo, hi, color) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        ax.fill_between(freqs, psd, where=mask, alpha=0.3, color=color, label=name)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (uV^2 / Hz)")
    ax.set_title("EEG Frequency Bands (P4 channel, subject 1)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "freq_bands_result.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
