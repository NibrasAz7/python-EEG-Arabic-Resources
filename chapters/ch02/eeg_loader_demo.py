"""Demonstrate the shared EEG data loader function.

Shows how to load real EEG data from the local PhysioNet dataset
using the load_local_eeg() function from utils/eeg_loader.py.

Usage:
    python eeg_loader_demo.py
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"


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
    print(f"Duration: {eeg_data.shape[0] / 200:.1f} seconds")
    print(f"Sampling rate: 200 Hz")

    # Access first channel (P4)
    p4_signal = eeg_data[:, 0]
    print(f"P4 signal - mean: {p4_signal.mean():.2f} uV")
    print(f"P4 signal - std:  {p4_signal.std():.2f} uV")


if __name__ == "__main__":
    main()
