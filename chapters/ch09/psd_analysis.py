"""Power Spectral Density analysis using Welch's method.

Computes PSD for all 4 channels of the local EEG dataset using
scipy.signal.welch, and visualizes the spectra alongside band
power distribution for channel P4.

Usage:
    python psd_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
NPERSEG = 1024
NOVERLAP = 512

BANDS = [
    ('Delta', 0.5, 4, '#2ca02c'),
    ('Theta', 4, 8, '#1f77b4'),
    ('Alpha', 8, 13, '#ff7f0e'),
    ('Beta', 13, 30, '#d62728'),
    ('Gamma', 30, 80, '#9467bd'),
]


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(4):
        freqs, psd = welch(eeg_data[:, i], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        axes[0].semilogy(freqs, psd, linewidth=1, color=colors[i], label=ch_names[i])

    for name, fmin, fmax, color in BANDS:
        axes[0].axvspan(fmin, fmax, alpha=0.08, color=color)

    axes[0].set_xlim(0, 80)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD (uV^2/Hz)')
    axes[0].set_title('Power Spectral Density - All channels (Welch)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    freqs, psd = welch(eeg_data[:, 0], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    band_powers = []
    band_colors = []
    for name, fmin, fmax, color in BANDS:
        mask = (freqs >= fmin) & (freqs <= fmax)
        power = np.trapezoid(psd[mask], freqs[mask])
        band_powers.append(power)
        band_colors.append(color)

    bar_positions = np.arange(len(BANDS))
    bar_names = [b[0] for b in BANDS]
    axes[1].bar(bar_positions, band_powers, color=band_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(bar_positions)
    axes[1].set_xticklabels(bar_names)
    axes[1].set_xlabel('Frequency band')
    axes[1].set_ylabel('Absolute power (uV^2)')
    axes[1].set_title('Band power distribution - Channel P4')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Power Spectral Density Analysis', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'psd_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
