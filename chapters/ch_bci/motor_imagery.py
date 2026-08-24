"""Visualize ERD/ERS in motor imagery data from BNCI2014-001.

Loads motor imagery data for subject 1, computes the band power in
the mu (8-12 Hz) and beta (13-30 Hz) bands over C3 and C4 channels,
and plots the time course of power during left vs right hand imagery.

Usage:
    python motor_imagery.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250


def band_power(epoch, fs, fmin, fmax):
    """Compute average band power for a single epoch (channels x samples)."""
    from scipy.signal import welch
    n_channels, n_samples = epoch.shape
    powers = np.zeros(n_channels)
    for ch in range(n_channels):
        freqs, psd = welch(epoch[ch, :], fs=fs, nperseg=min(256, n_samples))
        mask = (freqs >= fmin) & (freqs <= fmax)
        powers[ch] = np.trapezoid(psd[mask], freqs[mask])
    return powers


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]

    raw = dataset.get_data(subjects=[1])
    s1 = raw[1]
    sess = list(s1.values())[0]
    run = list(sess.values())[0]
    ch_names = run.ch_names
    print(f"Channels: {ch_names}")
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(labels)}")

    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    cz_idx = ch_names.index('Cz')

    mu_power_left = []
    mu_power_right = []
    beta_power_left = []
    beta_power_right = []

    for i, label in enumerate(labels):
        mu = band_power(X[i], FS, 8, 12)
        beta = band_power(X[i], FS, 13, 30)
        if label == 'left_hand':
            mu_power_left.append([mu[c3_idx], mu[cz_idx], mu[c4_idx]])
            beta_power_left.append([beta[c3_idx], beta[cz_idx], beta[c4_idx]])
        else:
            mu_power_right.append([mu[c3_idx], mu[cz_idx], mu[c4_idx]])
            beta_power_right.append([beta[c3_idx], beta[cz_idx], beta[c4_idx]])

    mu_power_left = np.array(mu_power_left)
    mu_power_right = np.array(mu_power_right)
    beta_power_left = np.array(beta_power_left)
    beta_power_right = np.array(beta_power_right)

    channels = ['C3', 'Cz', 'C4']
    mu_left_mean = mu_power_left.mean(axis=0)
    mu_right_mean = mu_power_right.mean(axis=0)
    beta_left_mean = beta_power_left.mean(axis=0)
    beta_right_mean = beta_power_right.mean(axis=0)

    print(f"\nMu band (8-12 Hz) average power:")
    print(f"  Left hand:  C3={mu_left_mean[0]:.2f}, Cz={mu_left_mean[1]:.2f}, C4={mu_left_mean[2]:.2f}")
    print(f"  Right hand: C3={mu_right_mean[0]:.2f}, Cz={mu_right_mean[1]:.2f}, C4={mu_right_mean[2]:.2f}")
    print(f"\nBeta band (13-30 Hz) average power:")
    print(f"  Left hand:  C3={beta_left_mean[0]:.2f}, Cz={beta_left_mean[1]:.2f}, C4={beta_left_mean[2]:.2f}")
    print(f"  Right hand: C3={beta_right_mean[0]:.2f}, Cz={beta_right_mean[1]:.2f}, C4={beta_right_mean[2]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(channels))
    width = 0.35
    axes[0].bar(x - width/2, mu_left_mean, width, label='Left hand', color='steelblue', edgecolor='black')
    axes[0].bar(x + width/2, mu_right_mean, width, label='Right hand', color='coral', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].set_ylabel('Power (V^2/Hz)')
    axes[0].set_title('Mu band (8-12 Hz) - ERD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - width/2, beta_left_mean, width, label='Left hand', color='steelblue', edgecolor='black')
    axes[1].bar(x + width/2, beta_right_mean, width, label='Right hand', color='coral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_ylabel('Power (V^2/Hz)')
    axes[1].set_title('Beta band (13-30 Hz) - ERD')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('ERD/ERS in Motor Imagery - BNCI2014-001 Subject 1', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'motor_imagery_result.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    left_avg = X[labels == 'left_hand'].mean(axis=0)
    right_avg = X[labels == 'right_hand'].mean(axis=0)
    t = np.arange(X.shape[2]) / FS
    ax.plot(t, left_avg[c3_idx], label='Left hand - C3', color='steelblue')
    ax.plot(t, right_avg[c4_idx], label='Right hand - C4', color='coral')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title('Averaged EEG signal at contralateral electrodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'motor_imagery_signal.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
