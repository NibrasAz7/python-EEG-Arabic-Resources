"""FFT-based evaluation of ICA artifact removal.

Compares the frequency spectrum of the P4 channel before and
after ICA cleaning (excluding the highest-variance component)
to evaluate the effect of artifact removal on brain wave bands.

Usage:
    python eval_fft.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import mne

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200

BANDS = [
    ("Delta", 0.5, 4, "#2ca02c"),
    ("Theta", 4, 8, "#1f77b4"),
    ("Alpha", 8, 13, "#ff7f0e"),
    ("Beta", 13, 30, "#d62728"),
    ("Gamma", 30, 80, "#9467bd"),
]


def compute_spectrum(data, fs):
    spectrum = fft(data)
    freqs = fftfreq(len(data), 1 / fs)
    magnitude = np.abs(spectrum)
    pos_mask = freqs >= 0
    return freqs[pos_mask], magnitude[pos_mask]


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    info = mne.create_info(ch_names, sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.T * 1e-6, info, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=4, random_state=97, max_iter=800, verbose=False
    )
    ica.fit(raw, verbose=False)

    component_variances = np.var(ica.get_sources(raw).get_data(), axis=1)
    exclude_idx = int(np.argmax(component_variances))
    ica.exclude = [exclude_idx]
    cleaned_raw = ica.apply(raw.copy(), verbose=False)
    cleaned_data = cleaned_raw.get_data()[0] * 1e6

    original_data = eeg_data[:, 0]

    freqs_orig, mag_orig = compute_spectrum(original_data, FS)
    freqs_clean, mag_clean = compute_spectrum(cleaned_data, FS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for name, fmin, fmax, color in BANDS:
        axes[0].axvspan(fmin, fmax, alpha=0.1, color=color)
        axes[1].axvspan(fmin, fmax, alpha=0.1, color=color)

    axes[0].plot(freqs_orig, mag_orig, linewidth=0.8, color='black')
    axes[0].set_xlim(0, 80)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('FFT Before ICA Cleaning - Channel P4')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs_clean, mag_clean, linewidth=0.8, color='green')
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('FFT After ICA Cleaning - Channel P4')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('FFT Evaluation - Before vs After ICA Cleaning', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'eval_fft_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
