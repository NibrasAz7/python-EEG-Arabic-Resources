"""Real-time sliding buffer processing simulation.

Simulates real-time EEG processing by iterating through the P4 channel
in chunks of 50 samples, maintaining a sliding buffer of 500 samples
(2.5 seconds), and applying a moving average filter at each step.

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
CHUNK_SIZE = 50
BUFFER_SIZE = 500
MA_WINDOW = 11
N_PLOT = 5000
CURRENT_POS = 2500


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered moving average filter to a 1D signal."""
    if len(data) < window:
        return data.copy()
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    signal = eeg_data[:, 0]

    n_samples = min(N_PLOT, len(signal))
    signal_plot = signal[:n_samples]

    buffer = np.zeros(BUFFER_SIZE)
    buffer_fill = 0
    filtered_output = np.zeros(n_samples)

    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        chunk = signal_plot[start:end]

        for i, sample in enumerate(chunk):
            if buffer_fill < BUFFER_SIZE:
                buffer[buffer_fill] = sample
                buffer_fill += 1
            else:
                buffer = np.roll(buffer, -1)
                buffer[-1] = sample

            filtered_buffer = moving_average(buffer[:buffer_fill], MA_WINDOW)
            filtered_output[start + i] = filtered_buffer[-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(signal_plot, color="blue", linewidth=0.5)
    axes[0].axvline(x=CURRENT_POS, color="black", linestyle="--", linewidth=1.5, label="Current position")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].set_title("Original Signal (P4 channel)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(filtered_output, color="red", linewidth=0.5, label="Real-time filtered output")
    axes[1].axvline(x=CURRENT_POS, color="black", linestyle="--", linewidth=1.5, label="Current position")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Amplitude (uV)")
    axes[1].set_title("Real-time Filtered Output (Moving Average, window=11)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Real-time Buffer Processing - Sliding Window", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "realtime_buffer_result.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
