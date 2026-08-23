"""Visualize electrode positions on a scalp topomap.

Loads the BNCI2014-001 motor imagery dataset (subject 1), extracts the
channel positions from the raw MNE info, and plots the electrode
positions on a 2D topomap with a circle representing the head and
labeled dots for each of the 22 electrodes.

Usage:
    python plot_cap.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from moabb.datasets import BNCI2014_001


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    ds = BNCI2014_001()
    sessions = ds.get_data(subjects=[1])
    first_run = next(iter(next(iter(next(iter(sessions.values())).values())).values()))

    montage = first_run.get_montage()
    ch_pos = montage.get_positions()['ch_pos']
    ch_names = [ch for ch in first_run.ch_names if ch in ch_pos]
    pos = np.array([ch_pos[ch] for ch in ch_names])

    fig, ax = plt.subplots(figsize=(8, 8))
    head = plt.Circle((0, 0), 1.0, color='white', ec='black', linewidth=2)
    ax.add_patch(head)
    nose = plt.Polygon([[0, 1.0], [-0.06, 1.1], [0.06, 1.1]], color='black')
    ax.add_patch(nose)

    pos_2d = pos[:, :2]
    scale = 1.0 / np.max(np.abs(pos_2d))
    pos_2d = pos_2d * scale * 0.95

    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], s=120, c='#1f77b4',
               edgecolors='black', zorder=5)
    for i, name in enumerate(ch_names):
        ax.annotate(name, (pos_2d[i, 0], pos_2d[i, 1]),
                    fontsize=7, ha='center', va='center', color='white',
                    fontweight='bold')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Electrode Positions - BNCI2014-001 (22 channels)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / 'plot_cap_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
