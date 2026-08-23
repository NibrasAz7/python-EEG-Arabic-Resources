"""Continuous Wavelet Transform (CWT) of EEG signals.

Applies CWT with Morlet wavelet to the P4 channel of the local
auditory EEG dataset and plots a scalogram showing how frequency
content changes over time.

Usage:
    python wavelet_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import pywt

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
FREQ_MIN = 0.5
FREQ_MAX = 80
NUM_FREQS = 100


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:N_PLOT, 0]

    freqs = np.linspace(FREQ_MIN, FREQ_MAX, NUM_FREQS)
    scales = pywt.frequency2scale("cmor1.5-1.0", freqs / FS)

    coefficients, _ = pywt.cwt(channel_data, scales, "cmor1.5-1.0")
    cwt_magnitude = np.abs(coefficients)

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='gray')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    im = axes[1].pcolormesh(
        time_sec, freqs, cwt_magnitude,
        shading='auto', cmap='viridis'
    )
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Wavelet Scalogram (CWT) - Channel P4')
    plt.colorbar(im, ax=axes[1], label='Magnitude')

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'wavelet_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
