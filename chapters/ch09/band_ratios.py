"""Band power ratios for mental state assessment.

Computes Theta/Alpha, Alpha/Beta, and Theta/Beta ratios on sliding
windows of the P4 channel to track mental state indicators over time.

Usage:
    python band_ratios.py
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
WINDOW = 1000
STEP = 500

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}


def compute_band_powers(segment, fs):
    freqs, psd = welch(segment, fs=fs, nperseg=1024)
    powers = {}
    for name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        powers[name] = np.trapezoid(psd[mask], freqs[mask])
    return powers


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_windows = (len(channel_data) - WINDOW) // STEP + 1
    theta_alpha = []
    alpha_beta = []
    theta_beta = []
    window_centers = []

    for i in range(n_windows):
        start = i * STEP
        end = start + WINDOW
        segment = channel_data[start:end]
        powers = compute_band_powers(segment, FS)

        theta_alpha.append(powers['theta'] / powers['alpha'])
        alpha_beta.append(powers['alpha'] / powers['beta'])
        theta_beta.append(powers['theta'] / powers['beta'])
        window_centers.append((start + end) / 2 / FS)

    theta_alpha = np.array(theta_alpha)
    alpha_beta = np.array(alpha_beta)
    theta_beta = np.array(theta_beta)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(window_centers, theta_alpha, linewidth=1.5, color='red')
    axes[0].axhline(y=np.mean(theta_alpha), color='black', linestyle='--', linewidth=1, label='Mean')
    axes[0].set_ylabel('Theta / Alpha')
    axes[0].set_title('Stress indicator (Theta/Alpha ratio)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(window_centers, alpha_beta, linewidth=1.5, color='green')
    axes[1].axhline(y=np.mean(alpha_beta), color='black', linestyle='--', linewidth=1, label='Mean')
    axes[1].set_ylabel('Alpha / Beta')
    axes[1].set_title('Relaxation indicator (Alpha/Beta ratio)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(window_centers, theta_beta, linewidth=1.5, color='blue')
    axes[2].axhline(y=np.mean(theta_beta), color='black', linestyle='--', linewidth=1, label='Mean')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Theta / Beta')
    axes[2].set_title('Attention indicator (Theta/Beta ratio)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Band Power Ratios - Mental State Indicators', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'band_ratios_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
