"""Visualize error-related negativity (ErrP) from ErpCore2021-ERN.

Loads ERN data for subject 1, computes the averaged ERP for Target
(error) and NonTarget (correct) trials at FCz and Cz, and shows the
characteristic negative deflection around 80-150 ms post-stimulus.

Usage:
    python errp.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import ErpCore2021_ERN
from moabb.paradigms import P300

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 1024


def main() -> None:
    dataset = ErpCore2021_ERN()
    paradigm = P300(fmin=0.5, fmax=40)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(labels)}")
    print(f"Targets: {np.sum(labels == 'Target')}, NonTargets: {np.sum(labels == 'NonTarget')}")

    raw = dataset.get_data(subjects=[1])
    s1 = raw[1]
    sess = list(s1.values())[0]
    run = list(sess.values())[0]
    ch_names = run.ch_names
    print(f"Channels (first 16): {ch_names[:16]}")

    target_channels = ['FCz', 'Cz', 'Fz']
    ch_indices = [ch_names.index(ch) for ch in target_channels]

    target_epochs = X[labels == 'Target']
    nontarget_epochs = X[labels == 'NonTarget']

    target_avg = target_epochs.mean(axis=0)
    nontarget_avg = nontarget_epochs.mean(axis=0)

    n_samples = X.shape[2]
    t = np.arange(n_samples) / FS * 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, ch_name, ch_idx in zip(axes, target_channels, ch_indices):
        ax.plot(t, target_avg[ch_idx], label='Target (error)', color='coral', linewidth=2)
        ax.plot(t, nontarget_avg[ch_idx], label='NonTarget (correct)', color='steelblue', linewidth=1.5)
        ax.axvspan(80, 150, color='red', alpha=0.15, label='ERN window (80-150 ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(f'{ch_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Error-Related Negativity (ErrP) - ErpCore2021-ERN Subject 1', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'errp_result.png', dpi=150)
    plt.close()

    ern_window = (t >= 80) & (t <= 150)
    print(f"\nERN amplitude (80-150 ms window, averaged):")
    for ch_name, ch_idx in zip(target_channels, ch_indices):
        t_amp = target_avg[ch_idx, ern_window].mean()
        nt_amp = nontarget_avg[ch_idx, ern_window].mean()
        diff = t_amp - nt_amp
        print(f"  {ch_name}: Target={t_amp:.6f} V, NonTarget={nt_amp:.6f} V, Diff={diff:.6f} V")

    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = [target_avg[ch_idx, ern_window].mean() - nontarget_avg[ch_idx, ern_window].mean()
             for ch_idx in ch_indices]
    ax.bar(target_channels, diffs, color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
    ax.set_ylabel('Amplitude difference (V)')
    ax.set_title('ERN amplitude difference (Target - NonTarget, 80-150 ms)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'errp_amplitude.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
