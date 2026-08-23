"""Real-time bandpass filtering with state maintenance.

Demonstrates streaming bandpass filtering using scipy.signal.lfilter
with filter state (zi) maintenance between chunks, and compares
the result with offline filtfilt filtering.

Usage:
    python realtime_filter.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, lfilter_zi

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
CHUNK_SIZE = 50
LOWCUT = 0.5
HIGHCUT = 30
ORDER = 4


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:N_PLOT, 0]

    b, a = butter(ORDER, [LOWCUT / (FS / 2), HIGHCUT / (FS / 2)], btype='band')
    zi = lfilter_zi(b, a)
    zi = zi * channel_data[0]

    realtime_output = np.zeros(N_PLOT)
    for i in range(0, N_PLOT, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, N_PLOT)
        chunk = channel_data[i:end]
        filtered_chunk, zi = lfilter(b, a, chunk, zi=zi)
        realtime_output[i:end] = filtered_chunk

    offline_output = filtfilt(b, a, channel_data)

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='blue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_sec, realtime_output, linewidth=0.5, color='green', label='Real-time (lfilter)')
    axes[1].plot(time_sec, offline_output, linewidth=0.5, color='red',
                 linestyle='--', label='Offline (filtfilt)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].set_title('Real-time vs Offline bandpass filter (0.5-30 Hz)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Real-time Bandpass Filter - Streaming vs Offline', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'realtime_filter_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
