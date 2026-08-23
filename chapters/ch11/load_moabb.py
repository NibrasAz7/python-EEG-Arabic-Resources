"""Load a MOABB dataset and print its structure.

Loads the BNCI2014-001 motor imagery dataset (subject 1 only) via MOABB,
extracts epochs with the MotorImagery paradigm, prints the dataset
structure (sessions, runs, channels, sampling rate) and epoch shapes,
writes a text summary to moabb_info.txt, and saves a bar chart of the
number of trials per class.

Usage:
    python load_moabb.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    ds = BNCI2014_001()
    sessions = ds.get_data(subjects=[1])

    subject_key = list(sessions.keys())[0]
    session_dict = sessions[subject_key]
    n_sessions = len(session_dict)
    n_runs = len(next(iter(session_dict.values())))
    first_run = next(iter(next(iter(session_dict.values())).values()))
    n_channels = len(first_run.ch_names)
    sfreq = first_run.info['sfreq']
    ch_names = first_run.ch_names

    paradigm = MotorImagery(n_classes=2)
    X, labels, meta = paradigm.get_data(dataset=ds, subjects=[1])

    unique_labels, counts = np.unique(labels, return_counts=True)

    lines = []
    lines.append("MOABB BNCI2014-001 - Dataset Structure")
    lines.append("=" * 45)
    lines.append(f"Subject: 1")
    lines.append(f"Sessions: {n_sessions}")
    lines.append(f"Runs per session: {n_runs}")
    lines.append(f"Channels: {n_channels}")
    lines.append(f"Sampling rate: {sfreq} Hz")
    lines.append(f"Channel names: {ch_names}")
    lines.append("")
    lines.append("Epochs (MotorImagery, n_classes=2)")
    lines.append("-" * 45)
    lines.append(f"X shape: {X.shape}")
    lines.append(f"Labels shape: {labels.shape}")
    lines.append(f"Meta shape: {meta.shape}")
    lines.append(f"Unique labels: {list(unique_labels)}")
    lines.append(f"Trials per class:")
    for lab, cnt in zip(unique_labels, counts):
        lines.append(f"  {lab}: {cnt}")
    lines.append(f"Epoch samples: {X.shape[2]}")
    lines.append(f"Epoch duration: {X.shape[2] / sfreq:.2f} s")

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "moabb_info.txt").write_text(summary, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique_labels, counts, color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Class label')
    ax.set_ylabel('Number of trials')
    ax.set_title('Trials per class - BNCI2014-001 (subject 1)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, cnt in enumerate(counts):
        ax.text(i, cnt + 1, str(cnt), ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'load_moabb_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
