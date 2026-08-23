"""FFT analysis of EEG signals with brain wave band visualization.

Applies Fast Fourier Transform to the P4 channel of the local
auditory EEG dataset and plots the frequency spectrum with the
five brain wave bands (delta, theta, alpha, beta, gamma) highlighted.

Usage:
    python fft_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000

BANDS = [
    ("Delta", 0.5, 4, "#2ca02c"),
    ("Theta", 4, 8, "#1f77b4"),
    ("Alpha", 8, 13, "#ff7f0e"),
    ("Beta", 13, 30, "#d62728"),
    ("Gamma", 30, 80, "#9467bd"),
]


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    spectrum = fft(channel_data)
    freqs = fftfreq(len(channel_data), 1 / FS)
    magnitude = np.abs(spectrum)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data[:N_PLOT], linewidth=0.5, color='gray')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    for name, fmin, fmax, color in BANDS:
        axes[1].axvspan(fmin, fmax, alpha=0.1, color=color)
        axes[1].text((fmin + fmax) / 2, axes[1].get_ylim()[1] * 0.95,
                     name, ha='center', fontsize=8, color=color)

    axes[1].plot(freqs, magnitude, linewidth=0.8, color='black')
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('FFT Spectrum - Channel P4')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'fft_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
