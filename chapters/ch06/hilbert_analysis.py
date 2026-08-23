"""Hilbert transform for instantaneous amplitude envelope of EEG.

Applies the Hilbert transform to the P4 channel of the local
auditory EEG dataset to extract the instantaneous amplitude
envelope and plots it overlaid on the original signal.

Usage:
    python hilbert_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:N_PLOT, 0]

    analytic_signal = hilbert(channel_data)
    amplitude_envelope = np.abs(analytic_signal)

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='gray')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_sec, channel_data, linewidth=0.5, color='blue', alpha=0.5, label='Signal')
    axes[1].plot(time_sec, amplitude_envelope, linewidth=1.5, color='red', label='Envelope')
    axes[1].plot(time_sec, -amplitude_envelope, linewidth=1.5, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].set_title('Hilbert Transform - Amplitude Envelope - Channel P4')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'hilbert_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
