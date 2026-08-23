"""Real-time bandpass filtering with streaming data.

Demonstrates real-time Butterworth bandpass filtering (0.5-30 Hz)
using scipy.signal.lfilter with state maintenance between chunks,
and compares it to offline filtfilt filtering.

Usage:
    python realtime_filter.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
LOWCUT = 0.5
HIGHCUT = 30.0
ORDER = 4
CHUNK_SIZE = 50
N_PLOT = 5000


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_samples = min(N_PLOT, len(channel_data))
    signal_plot = channel_data[:n_samples]

    nyq = 0.5 * FS
    b, a = signal.butter(ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band", analog=False)

    zi = signal.lfilter_zi(b, a)
    state = zi * signal_plot[0]
    realtime_filtered = np.zeros(n_samples)

    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        chunk = signal_plot[start:end]
        filtered_chunk, state = signal.lfilter(b, a, chunk, zi=state)
        realtime_filtered[start:end] = filtered_chunk

    offline_filtered = signal.filtfilt(b, a, signal_plot)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(signal_plot, color="blue", linewidth=0.5)
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].set_title("Original Signal (P4 channel)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(realtime_filtered, color="green", linewidth=0.5, label="Real-time (lfilter)")
    axes[1].plot(offline_filtered, color="red", linewidth=0.5, linestyle="--", label="Offline (filtfilt)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Amplitude (uV)")
    axes[1].set_title(f"Bandpass Filtered ({LOWCUT}-{HIGHCUT} Hz, order {ORDER})")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Real-time Bandpass Filter - Streaming vs Offline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "realtime_filter_result.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
