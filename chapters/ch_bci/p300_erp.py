"""Visualize P300 ERP from BNCI2014-009.

Loads P300 data for subject 1, computes the averaged ERP for Target
and NonTarget stimuli at Fz, Cz, Pz, and plots the waveforms to
reveal the characteristic P300 positive peak around 300 ms.

Usage:
    python p300_erp.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_009
from moabb.paradigms import P300

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 256


def main() -> None:
    dataset = BNCI2014_009()
    paradigm = P300(fmin=1, fmax=24)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(labels)}")
    print(f"Targets: {np.sum(labels == 'Target')}, NonTargets: {np.sum(labels == 'NonTarget')}")

    raw = dataset.get_data(subjects=[1])
    s1 = raw[1]
    sess = list(s1.values())[0]
    run = list(sess.values())[0]
    ch_names = run.ch_names
    print(f"Channels: {ch_names}")

    target_channels = ['Fz', 'Cz', 'Pz']
    ch_indices = [ch_names.index(ch) for ch in target_channels]

    target_epochs = X[labels == 'Target']
    nontarget_epochs = X[labels == 'NonTarget']

    target_avg = target_epochs.mean(axis=0)
    nontarget_avg = nontarget_epochs.mean(axis=0)

    n_samples = X.shape[2]
    t = np.arange(n_samples) / FS * 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, ch_name, ch_idx in zip(axes, target_channels, ch_indices):
        ax.plot(t, target_avg[ch_idx], label='Target', color='coral', linewidth=2)
        ax.plot(t, nontarget_avg[ch_idx], label='NonTarget', color='steelblue', linewidth=1.5)
        ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5, label='P300 (~300 ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(f'{ch_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('P300 ERP - BNCI2014-009 Subject 1', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'p300_erp_result.png', dpi=150)
    plt.close()

    p300_window = (t >= 250) & (t <= 400)
    print(f"\nP300 amplitude (250-400 ms window, averaged):")
    for ch_name, ch_idx in zip(target_channels, ch_indices):
        t_amp = target_avg[ch_idx, p300_window].mean()
        nt_amp = nontarget_avg[ch_idx, p300_window].mean()
        print(f"  {ch_name}: Target={t_amp:.6f} V, NonTarget={nt_amp:.6f} V, Diff={t_amp - nt_amp:.6f} V")

    fig, ax = plt.subplots(figsize=(8, 5))
    peak_amps = [target_avg[ch_idx, p300_window].max() - nontarget_avg[ch_idx, p300_window].max()
                 for ch_idx in ch_indices]
    ax.bar(target_channels, peak_amps, color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
    ax.set_ylabel('Peak amplitude difference (V)')
    ax.set_title('P300 peak amplitude difference (Target - NonTarget)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'p300_peak.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
