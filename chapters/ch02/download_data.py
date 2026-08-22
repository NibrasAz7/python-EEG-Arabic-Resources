"""Download EEG data from PhysioNet for the book's examples.

Clones the resources repository and downloads subject 7 data
from the PhysioNet auditory-eeg dataset.

Usage:
    python download_data.py
"""

import os
from pathlib import Path


def main() -> None:
    # Clone the resources repository (if not already cloned)
    if not os.path.exists('python-EEG-Arabic-Resources'):
        os.system(
            'git clone https://github.com/NibrasAz7/'
            'python-EEG-Arabic-Resources.git'
        )

    os.chdir('python-EEG-Arabic-Resources')

    # Download subject 7 data
    data_dir = Path('data/local')
    if not data_dir.exists() or not any(data_dir.glob('*.dat')):
        os.system(
            'python data/download_local.py '
            '--output data/local --subjects 7'
        )

    # Verify download
    dat_files = list(data_dir.glob('s07_*.dat'))
    print(f"Downloaded {len(dat_files)} recordings for subject 7")
    for f in sorted(dat_files):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
