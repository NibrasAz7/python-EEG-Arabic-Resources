"""SNR evaluation of ICA artifact removal across channels.

Computes the signal-to-noise ratio (alpha band power vs. rest)
for all 4 channels before and after ICA cleaning, and compares
them in a grouped bar chart.

Usage:
    python eval_snr.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
ALPHA_MIN = 8
ALPHA_MAX = 13


def compute_snr(data, fs, fmin, fmax):
    freqs, psd = welch(data, fs=fs, nperseg=1024)
    signal_mask = (freqs >= fmin) & (freqs <= fmax)
    noise_mask = (freqs >= 0.5) & (freqs <= 80) & ~signal_mask
    signal_power = np.trapezoid(psd[signal_mask], freqs[signal_mask])
    noise_power = np.trapezoid(psd[noise_mask], freqs[noise_mask])
    if noise_power == 0:
        return 0.0
    return 10 * np.log10(signal_power / noise_power)


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )

    info = mne.create_info(ch_names, sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.T * 1e-6, info, verbose=False)

    # NOTE: ICA works best with more channels than components.
    # With only 4 channels, we use n_components=3 to allow separation.
    # Real ICA artifact removal typically uses 16-64+ channel montages.
    ica = mne.preprocessing.ICA(
        n_components=3, random_state=97, max_iter=800, verbose=False
    )
    ica.fit(raw, verbose=False)

    component_variances = np.var(ica.get_sources(raw).get_data(), axis=1)
    exclude_idx = int(np.argmax(component_variances))
    ica.exclude = [exclude_idx]
    cleaned_raw = ica.apply(raw.copy(), verbose=False)
    cleaned_data = cleaned_raw.get_data() * 1e6

    snr_before = [compute_snr(eeg_data[:, i], FS, ALPHA_MIN, ALPHA_MAX) for i in range(4)]
    snr_after = [compute_snr(cleaned_data[i], FS, ALPHA_MIN, ALPHA_MAX) for i in range(4)]

    x = np.arange(len(ch_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, snr_before, width, label='Before ICA', color='steelblue')
    ax.bar(x + width / 2, snr_after, width, label='After ICA', color='orange')
    ax.set_xlabel('Channel')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Evaluation - Before vs After ICA Cleaning')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / 'eval_snr_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
