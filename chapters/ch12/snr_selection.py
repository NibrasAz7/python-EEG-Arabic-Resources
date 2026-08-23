"""Channel selection by Signal-to-Noise Ratio (SNR).

Computes SNR for each of the 22 channels of BNCI2014-001 subject 1,
ranks channels by SNR, and visualizes the ranking alongside a
topomap showing selected vs rejected channels.

Usage:
    python snr_selection.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent
N_SELECT = 10


def compute_snr(signal):
    signal_var = np.var(signal)
    window = 50
    moving_avg = np.convolve(signal, np.ones(window) / window, mode='same')
    noise = signal - moving_avg
    noise_var = np.var(noise)
    if noise_var == 0:
        return 0.0
    return signal_var / noise_var


def get_channel_positions():
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
    return ch_names, pos_2d


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    n_trials, n_channels, n_samples = X.shape
    snr_values = np.zeros(n_channels)
    for ch in range(n_channels):
        snr_list = [compute_snr(X[trial, ch, :]) for trial in range(n_trials)]
        snr_values[ch] = np.mean(snr_list)

    sorted_idx = np.argsort(snr_values)[::-1]
    selected = sorted_idx[:N_SELECT]

    ch_names, pos_2d = get_channel_positions()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['green' if i in selected else 'gray' for i in range(n_channels)]
    axes[0].bar(range(n_channels), snr_values[sorted_idx], color=colors, edgecolor='black')
    axes[0].set_xticks(range(n_channels))
    axes[0].set_xticklabels([ch_names[i] for i in sorted_idx], rotation=90, fontsize=7)
    axes[0].set_xlabel('Channel (sorted by SNR)')
    axes[0].set_ylabel('SNR value')
    axes[0].set_title('Channel SNR ranking')
    axes[0].grid(True, alpha=0.3, axis='y')

    head = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    axes[1].add_patch(head)
    nose = plt.Polygon([[0, 1.0], [-0.08, 1.12], [0.08, 1.12]], fill=False, color='black', linewidth=1.5)
    axes[1].add_patch(nose)
    for i in range(n_channels):
        color = 'green' if i in selected else 'red'
        axes[1].scatter(pos_2d[i, 0], pos_2d[i, 1], c=color, s=100, zorder=5, edgecolors='black')
        axes[1].annotate(ch_names[i], (pos_2d[i, 0], pos_2d[i, 1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Selected (green) vs rejected (red) channels')

    plt.suptitle('Channel Selection by SNR', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'snr_selection_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
