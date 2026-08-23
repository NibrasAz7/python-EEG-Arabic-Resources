"""LSL (Lab Streaming Layer) stream outlet and inlet demo.

Creates a simulated EEG stream outlet using pylsl, pushes 4-channel
EEG data in chunks, then creates an inlet to receive the data back.
Falls back to a simple copy simulation if pylsl is unavailable.

Usage:
    python lsl_stream.py
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
N_PLOT = 5000
CHANNEL_COUNT = 4


def run_lsl_stream(eeg_data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Push data through an LSL outlet and receive it via an inlet.

    Returns a tuple of (received_data, used_real_lsl).
    """
    from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

    info = StreamInfo(
        name="SimulatedEEG",
        type="EEG",
        channel_count=CHANNEL_COUNT,
        sampling_rate=FS,
        channel_format="float32",
    )
    outlet = StreamOutlet(info)

    n_samples = eeg_data.shape[0]
    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        chunk = eeg_data[start:end].astype(np.float32)
        outlet.push_chunk(chunk.tolist())

    import time
    time.sleep(0.5)

    streams = resolve_byprop("name", "SimulatedEEG", timeout=2)
    inlet = StreamInlet(streams[0])

    received = []
    total_received = 0
    while total_received < n_samples:
        chunk_data, timestamps_chunk = inlet.pull_chunk(timeout=1.0)
        if not chunk_data:
            break
        received.extend(chunk_data)
        total_received += len(chunk_data)

    received = np.array(received[:n_samples])
    return received, True


def run_simulated_stream(eeg_data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Simulate LSL by copying data directly.

    Returns a tuple of (received_data, used_real_lsl).
    """
    return eeg_data.copy(), False


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    try:
        received_data, used_real_lsl = run_lsl_stream(eeg_data)
        print("Real LSL stream used successfully.")
    except Exception as e:
        print(f"LSL not available ({e}), falling back to simulation.")
        received_data, used_real_lsl = run_simulated_stream(eeg_data)
        print("Simulation mode used (data copied directly).")

    n_plot = min(N_PLOT, eeg_data.shape[0])
    offsets = np.array([0, 100, 200, 300])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(CHANNEL_COUNT):
        axes[0].plot(eeg_data[:n_plot, i] + offsets[i], linewidth=0.5, label=ch_names[i])
    axes[0].set_ylabel("Amplitude + offset (uV)")
    axes[0].set_title("Original 4-Channel Signal")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i in range(CHANNEL_COUNT):
        axes[1].plot(received_data[:n_plot, i] + offsets[i], linewidth=0.5, label=ch_names[i])
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Amplitude + offset (uV)")
    axes[1].set_title("Received 4-Channel Signal via LSL" + (" (Simulation)" if not used_real_lsl else ""))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("LSL Stream - Outlet and Inlet Demo", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "lsl_stream_result.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
