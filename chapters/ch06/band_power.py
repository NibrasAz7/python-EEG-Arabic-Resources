"""Brain wave band power analysis using Welch's method.

Computes the power spectral density of the P4 channel using
Welch's method, then integrates the power in each brain wave
band (delta, theta, alpha, beta, gamma) and compares them.
Applies a 1-45 Hz bandpass and 50 Hz notch filter to clean
the raw signal before analysis.

Usage:
    python band_power.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, iirnotch

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


def clean_signal(signal, fs=FS, low=1.0, high=45.0, notch_freq=50.0):
    b_bp, a_bp = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    filtered = filtfilt(b_bp, a_bp, signal)
    b_notch, a_notch = iirnotch(notch_freq, 30.0, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]
    channel_data = clean_signal(channel_data)

    freqs, psd = welch(channel_data, fs=FS, nperseg=1024)

    band_powers = {}
    for name, fmin, fmax, color in BANDS:
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        power = np.trapezoid(psd[band_mask], freqs[band_mask])
        band_powers[name] = power

    total_power = sum(band_powers.values())
    relative_powers = {k: v / total_power * 100 for k, v in band_powers.items()}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for name, fmin, fmax, color in BANDS:
        axes[0].axvspan(fmin, fmax, alpha=0.1, color=color)
        axes[0].text((fmin + fmax) / 2, axes[0].get_ylim()[1] * 0.9,
                     name, ha='center', fontsize=8, color=color)
    axes[0].semilogy(freqs, psd, linewidth=1, color='black')
    axes[0].set_xlim(0, 80)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD (uV^2/Hz)')
    axes[0].set_title('Power Spectral Density - Channel P4')
    axes[0].grid(True, alpha=0.3)

    names = list(relative_powers.keys())
    values = list(relative_powers.values())
    colors = [c[3] for c in BANDS]
    axes[1].bar(names, values, color=colors)
    axes[1].set_ylabel('Relative Power (%)')
    axes[1].set_title('Relative Band Power - Channel P4')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'band_power_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
