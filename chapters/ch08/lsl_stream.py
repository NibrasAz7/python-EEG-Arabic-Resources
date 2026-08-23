"""LSL stream outlet and inlet demonstration with EEG data.

Creates a simulated LSL outlet, pushes 4-channel EEG data from the
local dataset, then creates an inlet to receive the data back.
Falls back to a simple copy simulation if LSL is not available.

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
N_PLOT = 5000
CHUNK_SIZE = 50
CHANNEL_COUNT = 4


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    use_lsl = False
    received_data = None

    try:
        from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
        import time

        info = StreamInfo(
            name='SimulatedEEG', type='EEG',
            channel_count=CHANNEL_COUNT, nominal_srate=FS,
            channel_format='float32'
        )
        outlet = StreamOutlet(info)
        time.sleep(0.5)

        n_samples = eeg_data.shape[0]
        for i in range(0, n_samples, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, n_samples)
            chunk = eeg_data[i:end, :].astype(np.float32)
            outlet.push_chunk(chunk)

        inlets = resolve_byprop('name', 'SimulatedEEG', timeout=2.0)
        if len(inlets) > 0:
            inlet = StreamInlet(inlets[0])
            received = []
            attempts = 0
            while len(received) < n_samples and attempts < 100:
                samples, _ = inlet.pull_chunk(timeout=0.5)
                if samples:
                    received.extend(samples)
                else:
                    attempts += 1
            if len(received) >= n_samples:
                received_data = np.array(received[:n_samples])
                use_lsl = True
    except Exception:
        pass

    if not use_lsl:
        received_data = eeg_data.copy()

    time_sec = np.arange(N_PLOT) / FS
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    mode_label = 'LSL' if use_lsl else 'Simulation'

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for i in range(4):
        offset = i * 200
        axes[0].plot(time_sec, eeg_data[:N_PLOT, i] + offset,
                     linewidth=0.5, color=colors[i], label=ch_names[i])
    axes[0].set_yticks([])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Channels')
    axes[0].set_title('Original EEG (Outlet)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    for i in range(4):
        offset = i * 200
        axes[1].plot(time_sec, received_data[:N_PLOT, i] + offset,
                     linewidth=0.5, color=colors[i], label=ch_names[i])
    axes[1].set_yticks([])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Channels')
    axes[1].set_title(f'Received EEG (Inlet) - Mode: {mode_label}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LSL Stream - Outlet and Inlet Demo', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'lsl_stream_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
