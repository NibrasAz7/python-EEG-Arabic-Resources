"""Electrode position visualization for MOABB dataset.

Plots the 22-channel electrode positions of BNCI2014-001 on a
2D scalp topomap using MNE's sensor plotting capabilities.

Usage:
    python plot_cap.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[1])
    subject_data = data[1]
    session_key = list(subject_data.keys())[0]
    run_key = list(subject_data[session_key].keys())[0]
    raw = subject_data[session_key][run_key]

    montage = raw.get_montage()
    ch_pos = montage.get_positions()['ch_pos']
    ch_names = [ch for ch in raw.ch_names if ch in ch_pos and not ch.startswith('EOG')]
    positions = np.array([ch_pos[ch] for ch in ch_names])
    pos_2d = positions[:, :2]

    scale = 1.0 / np.max(np.abs(pos_2d))
    pos_2d = pos_2d * scale * 0.95

    fig, ax = plt.subplots(figsize=(8, 8))
    head_circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax.add_patch(head_circle)

    nose = plt.Polygon([[0, 1.0], [-0.08, 1.12], [0.08, 1.12]], fill=False, color='black', linewidth=1.5)
    ax.add_patch(nose)

    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='red', s=80, zorder=5, edgecolors='black')
    for i, name in enumerate(ch_names):
        ax.annotate(name, (pos_2d[i, 0], pos_2d[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Electrode Positions - BNCI2014-001 (22 channels)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_cap_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
