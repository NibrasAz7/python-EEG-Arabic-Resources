"""Real-time buffer processing simulation on EEG signals.

Simulates real-time processing by iterating through the P4 channel
in chunks, maintaining a sliding buffer, and applying a moving
average filter at each step to demonstrate streaming processing.

Usage:
    python realtime_buffer.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
N_PLOT = 5000
BUFFER_SIZE = 500
CHUNK_SIZE = 50
WINDOW = 11


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:N_PLOT, 0]

    filtered_output = np.zeros(N_PLOT)
    buffer = np.zeros(BUFFER_SIZE)

    for i in range(0, N_PLOT, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, N_PLOT)
        for j in range(i, end):
            buffer = np.roll(buffer, -1)
            buffer[-1] = channel_data[j]
            kernel = np.ones(WINDOW) / WINDOW
            filtered = np.convolve(buffer, kernel, mode='same')
            filtered_output[j] = filtered[BUFFER_SIZE // 2]

    time_sec = np.arange(N_PLOT) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='blue')
    axes[0].axvline(x=2500 / FS, color='red', linestyle='--', linewidth=1, label='Current position')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title('Original signal - Channel P4')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_sec, filtered_output, linewidth=0.5, color='red')
    axes[1].axvline(x=2500 / FS, color='red', linestyle='--', linewidth=1, label='Current position')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].set_title('Real-time filtered output (sliding buffer)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Real-time Buffer Processing - Sliding Window', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'realtime_buffer_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
