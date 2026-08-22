"""Load EEG data from the local PhysioNet dataset and display basic info.

Demonstrates using load_local_eeg() to read WFDB files and inspect
the signal shape, duration, and per-channel statistics.

Usage:
    python load_eeg_data.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    # Load subject 7, experiment 1, session 2
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR,
        subject=7,
        experiment=1,
        session=2
    )

    print(f"Signals shape: {eeg_data.shape}")
    print(f"Channels: {ch_names}")
    print(f"Duration: {eeg_data.shape[0] / FS:.1f} seconds")
    print(f"Sampling rate: {FS} Hz")

    # Basic statistics per channel
    for i, ch in enumerate(ch_names):
        signal = eeg_data[:, i]
        print(f"  {ch}: mean={signal.mean():.2f} uV, "
              f"std={signal.std():.2f} uV, "
              f"range=[{signal.min():.1f}, {signal.max():.1f}] uV")


if __name__ == "__main__":
    main()
