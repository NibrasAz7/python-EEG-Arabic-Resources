"""Convert raw EEG data to an MNE-Python Raw object.

Demonstrates creating an mne.io.RawArray from data loaded via
load_local_eeg(), including channel names and sampling rate.

Usage:
    python create_raw.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    # Create MNE info structure
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=FS,
        ch_types='eeg'
    )

    # Create Raw object (MNE expects data in shape: n_channels x n_times)
    raw = mne.io.RawArray(eeg_data.T, info)

    print(raw)
    print(f"Channels: {raw.ch_names}")
    print(f"Duration: {raw.times[-1]:.1f} s")


if __name__ == "__main__":
    main()
