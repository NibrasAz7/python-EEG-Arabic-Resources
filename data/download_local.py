"""Download the local auditory EEG dataset from PhysioNet.

Dataset: auditory-eeg/1.0.0
Author: Nibras Abu Al-Dhahab
URL: https://physionet.org/content/auditory-eeg/1.0.0/

Usage:
    python download_local.py --output ./local_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve


PHYSIONET_BASE = "https://physionet.org/files/auditory-eeg/1.0.0/"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading: {dest.name} ...", end=" ", flush=True)
    urlretrieve(url, dest)
    print("done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download auditory EEG data from PhysioNet")
    parser.add_argument("--output", type=str, default="./local_data",
                        help="Output directory (default: ./local_data)")
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="Subject numbers to download (default: all 20)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, 21))

    print(f"Downloading auditory EEG data to: {output_dir}")
    print(f"Subjects: {subjects}")
    print(f"Source: {PHYSIONET_BASE}\n")

    # Download RECORDS.txt first to get file list
    records_url = PHYSIONET_BASE + "RECORDS.txt"
    records_path = output_dir / "RECORDS.txt"
    download_file(records_url, records_path)

    # Read file list
    with open(records_path) as f:
        files = [line.strip() for line in f if line.strip()]

    # Filter by requested subjects
    def matches_subject(filename: str, subj: int) -> bool:
        return filename.startswith(f"s{subj:02d}_")

    if args.subjects:
        files = [f for f in files
                 if any(matches_subject(f, s) for s in args.subjects)]

    print(f"\nDownloading {len(files)} files...\n")
    for i, filename in enumerate(files, 1):
        dest = output_dir / filename
        if dest.exists():
            print(f"  [{i}/{len(files)}] {filename} (already exists, skipping)")
            continue
        url = PHYSIONET_BASE + filename
        download_file(url, dest)

    print(f"\nDone. {len(files)} files in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
