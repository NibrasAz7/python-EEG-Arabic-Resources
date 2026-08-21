"""Shared EEG data loading utilities for the Arabic EEG book.

This module provides functions to load:
1. Local auditory EEG data (PhysioNet, 4 channels, 20 subjects)
2. MOABB datasets (motor imagery, P300, SSVEP)

Usage:
    from utils.eeg_loader import load_local_eeg, load_moabb_dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Local dataset constants
LOCAL_CHANNELS = ["P4", "Cz", "F8", "T7"]
LOCAL_SAMPLING_RATE = 1000  # Hz
LOCAL_N_SUBJECTS = 20
LOCAL_N_EXPERIMENTS = 10


def load_local_eeg(
    data_dir: str | Path,
    subject: int,
    experiment: int,
    session: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a single recording from the local auditory EEG dataset.

    Args:
        data_dir: Path to the directory containing CSV files.
        subject: Subject number (1-20).
        experiment: Experiment number (1-10).
        session: Optional session number (1-3) for multi-session experiments.

    Returns:
        Tuple of (timestamps, signals, channel_names) where:
        - timestamps: 1D array of sample timestamps (ms)
        - signals: 2D array of shape (n_samples, n_channels)
        - channel_names: list of channel names
    """
    data_dir = Path(data_dir)

    if session is not None:
        filename = f"s{subject:02d}_ex{experiment:02d}_s{session:02d}.csv"
    else:
        filename = f"s{subject:02d}_ex{experiment:02d}.csv"

    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"EEG file not found: {filepath}")

    df = pd.read_csv(filepath)
    timestamps = df.iloc[:, 0].values
    signals = df.iloc[:, 1:].values
    channel_names = list(df.columns[1:])

    return timestamps, signals, channel_names


def load_local_subject_all(
    data_dir: str | Path,
    subject: int,
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Load all recordings for a single subject.

    Args:
        data_dir: Path to the directory containing CSV files.
        subject: Subject number (1-20).

    Returns:
        List of (timestamps, signals, channel_names) tuples.
    """
    data_dir = Path(data_dir)
    recordings = []

    # Experiments 1 and 2 have 3 sessions each
    for exp in [1, 2]:
        for sess in [1, 2, 3]:
            try:
                rec = load_local_eeg(data_dir, subject, exp, sess)
                recordings.append(rec)
            except FileNotFoundError:
                pass

    # Experiments 5-10 have single sessions
    for exp in [5, 6, 7, 8, 9, 10]:
        try:
            rec = load_local_eeg(data_dir, subject, exp)
            recordings.append(rec)
        except FileNotFoundError:
            pass

    return recordings


def load_moabb_dataset(
    dataset_name: str = "BNCI2014-001",
    subject_list: Optional[list[int]] = None,
):
    """Load a dataset from MOABB.

    Args:
        dataset_name: MOABB dataset name (e.g., "BNCI2014-001" for MI).
        subject_list: Optional list of subject numbers. None loads all.

    Returns:
        MOABB dataset object with data loaded.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset_map = {
        "BNCI2014-001": BNCI2014_001,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = dataset_map[dataset_name]()
    paradigm = MotorImagery()

    if subject_list is not None:
        dataset.subject_list = subject_list

    X, labels, meta = paradigm.get_data(dataset=dataset)
    return X, labels, meta


if __name__ == "__main__":
    # Quick test: load one file from local data
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        ts, sig, ch = load_local_eeg(data_dir, subject=1, experiment=1, session=1)
        print(f"Loaded: {sig.shape[0]} samples, {sig.shape[1]} channels: {ch}")
        print(f"Duration: {sig.shape[0] / LOCAL_SAMPLING_RATE:.2f} seconds")
    except FileNotFoundError as e:
        print(f"Could not load test file: {e}")
        print("Usage: python eeg_loader.py <path_to_csv_directory>")
