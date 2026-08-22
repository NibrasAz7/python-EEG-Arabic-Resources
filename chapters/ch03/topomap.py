"""Plot spatial distribution of brain activity on the scalp.

Uses MNE-Python to create a topographic map (topomap) showing
the power spectral density across scalp electrode positions.

Usage:
    python topomap.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import mne
from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200  # Sampling rate (Hz)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    # Create MNE Raw object with standard montage
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.T, info)

    # Assign standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Plot topomap of power spectral density
    fig = raw.compute_psd().plot_topomap()
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'eeg_topomap.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
