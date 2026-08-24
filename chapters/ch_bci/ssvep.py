"""Visualize SSVEP responses from Nakanishi2015.

Loads SSVEP data for subject 1, computes the power spectral density
at occipital channels (O1, Oz, O2), and shows peaks at the stimulus
frequencies and their harmonics.

Usage:
    python ssvep.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from moabb.datasets import Nakanishi2015
from moabb.paradigms import SSVEP

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 256


def main() -> None:
    dataset = Nakanishi2015()
    paradigm = SSVEP(fmin=7, fmax=45, n_classes=4)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    print(f"Data shape: {X.shape}")
    print(f"Labels (stimulus frequencies): {np.unique(labels)}")

    raw = dataset.get_data(subjects=[1])
    s1 = raw[1]
    sess = list(s1.values())[0]
    run = list(sess.values())[0]
    ch_names = run.ch_names
    print(f"Channels: {ch_names}")

    target_channels = ['O1', 'Oz', 'O2']
    ch_indices = [ch_names.index(ch) for ch in target_channels if ch in ch_names]
    target_channels = [ch_names[i] for i in ch_indices]

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    freqs_psd = None
    for i, label in enumerate(unique_labels):
        epochs = X[labels == label]
        avg_epoch = epochs.mean(axis=0)

        for ch_idx, ch_name in zip(ch_indices, target_channels):
            freqs, psd = welch(avg_epoch[ch_idx, :], fs=FS, nperseg=min(1024, avg_epoch.shape[1]))
            if freqs_psd is None:
                freqs_psd = freqs
            axes[i].semilogy(freqs, psd, label=ch_name, alpha=0.8)

        stim_freq = float(label)
        axes[i].axvline(x=stim_freq, color='red', linestyle='--', alpha=0.7, label=f'Stimulus ({stim_freq} Hz)')
        axes[i].axvline(x=stim_freq * 2, color='orange', linestyle=':', alpha=0.5, label=f'2nd harmonic')
        axes[i].set_xlim(5, 50)
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('PSD (V^2/Hz)')
        axes[i].set_title(f'Stimulus: {label} Hz')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle('SSVEP Power Spectral Density - Nakanishi2015 Subject 1', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ssvep_result.png', dpi=150)
    plt.close()

    print(f"\nSNR at stimulus frequency (Oz channel):")
    oz_idx = ch_names.index('Oz') if 'Oz' in ch_names else ch_indices[0]
    snr_results = []
    for label in unique_labels:
        epochs = X[labels == label]
        avg_epoch = epochs.mean(axis=0)
        freqs, psd = welch(avg_epoch[oz_idx, :], fs=FS, nperseg=min(1024, avg_epoch.shape[1]))
        stim_freq = float(label)
        mask_stim = (freqs >= stim_freq - 0.5) & (freqs <= stim_freq + 0.5)
        mask_noise = (freqs >= stim_freq - 3) & (freqs <= stim_freq + 3) & ~mask_stim
        signal_power = psd[mask_stim].mean()
        noise_power = psd[mask_noise].mean()
        snr = 10 * np.log10(signal_power / noise_power)
        snr_results.append((label, snr))
        print(f"  {label} Hz: SNR = {snr:.2f} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    freqs_str = [f"{l} Hz" for l, _ in snr_results]
    snrs = [s for _, s in snr_results]
    ax.bar(freqs_str, snrs, color='steelblue', edgecolor='black')
    ax.set_ylabel('SNR (dB)')
    ax.set_xlabel('Stimulus frequency')
    ax.set_title('SSVEP SNR at Oz channel')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ssvep_snr.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
