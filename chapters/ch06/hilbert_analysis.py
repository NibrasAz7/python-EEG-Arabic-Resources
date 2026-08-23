"""Hilbert transform for instantaneous amplitude envelope of EEG.

Applies the Hilbert transform to the P4 channel of the local
auditory EEG dataset to extract the instantaneous amplitude
envelope and plots it overlaid on the original signal.
Applies a 1-45 Hz bandpass and 50 Hz notch filter to clean
the raw signal before analysis.

Usage:
    python hilbert_analysis.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, iirnotch

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000


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
    channel_data = eeg_data[:N_PLOT, 0]
    channel_data = clean_signal(channel_data)

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
