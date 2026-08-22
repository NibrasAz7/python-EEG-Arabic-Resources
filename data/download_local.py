"""Download the local auditory EEG dataset from PhysioNet.

Dataset: auditory-eeg/1.0.0
Author: Nibras Abo Alzahab
URL: https://physionet.org/content/auditory-eeg/1.0.0/

The dataset is in WFDB format (.dat + .hea file pairs) organized under
WFDB_Files/Raw_Data/. This script downloads the .dat and .hea files
for the requested subjects.

Usage:
    python download_local.py --output ./data/local
    python download_local.py --output ./data/local --subjects 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve


PHYSIONET_BASE = "https://physionet.org/files/auditory-eeg/1.0.0/"
RAW_DATA_URL = PHYSIONET_BASE + "WFDB_Files/Raw_Data/"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading: {dest.name} ...", end=" ", flush=True)
    urlretrieve(url, dest)
    print("done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download auditory EEG data from PhysioNet")
    parser.add_argument("--output", type=str, default="./data/local",
                        help="Output directory (default: ./data/local)")
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="Subject numbers to download (default: all 20)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, 21))

    print(f"Downloading auditory EEG data to: {output_dir}")
    print(f"Subjects: {subjects}")
    print(f"Source: {RAW_DATA_URL}\n")

    # Build the file list: each subject has experiments 1-10
    # Experiments 1 and 2 have 3 sessions each (s01, s02, s03)
    # Experiments 5-10 have a single session
    # Experiments 3 and 4 (resting-state) are not available for all subjects
    recordings = []
    for subj in subjects:
        for exp in [1, 2]:
            for sess in [1, 2, 3]:
                recordings.append(f"s{subj:02d}_ex{exp:02d}_s{sess:02d}")
        for exp in [5, 6, 7, 8, 9, 10]:
            recordings.append(f"s{subj:02d}_ex{exp:02d}")

    print(f"Total recordings: {len(recordings)} (each has .dat + .hea)\n")

    downloaded = 0
    skipped = 0
    for i, rec in enumerate(recordings, 1):
        for ext in [".hea", ".dat"]:
            dest = output_dir / (rec + ext)
            if dest.exists():
                skipped += 1
                continue
            url = RAW_DATA_URL + rec + ext
            try:
                print(f"  [{i}/{len(recordings)}] {rec}{ext}", end=" ... ", flush=True)
                urlretrieve(url, dest)
                print("done")
                downloaded += 1
            except Exception as e:
                print(f"failed: {e}")
                # Remove partial file if any
                if dest.exists():
                    dest.unlink()

    print(f"\nDone. Downloaded: {downloaded}, skipped (already exist): {skipped}")
    print(f"Files in {output_dir}: {len(list(output_dir.glob('*.dat')))} recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
