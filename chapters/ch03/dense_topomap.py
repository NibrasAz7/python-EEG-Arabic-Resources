"""Plot topographic maps using the MNE sample dataset (60 EEG channels).

Demonstrates proper topographic visualization with a dense electrode
array. Downloads the MNE sample dataset, creates epochs around auditory
events, computes evoked responses, and plots topomaps at multiple
time points.

Usage:
    python dense_topomap.py
"""

import matplotlib.pyplot as plt
import mne


def main() -> None:
    # Download the MNE sample dataset (1.5 GB, one-time download)
    sample_path = mne.datasets.sample.data_path()

    # Load the EEG data (60 channels, pre-configured montage)
    raw_fname = sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    # Pick only EEG channels (exclude MEG and other channels)
    raw.pick_types(eeg=True)

    print(f"Number of EEG channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names[:10]} ...")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} s")

    # The montage is already set in the sample dataset
    print(f"Montage: {raw.get_montage()}")

    # Create epochs around auditory events (event ID 1 = standard tone)
    events = mne.find_events(raw, stim_channel='STI 014')
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                        preload=True)

    # Compute evoked response (average across epochs)
    evoked = epochs.average()

    # Plot topomap at multiple time points
    fig = evoked.plot_topomap(times=[0.0, 0.1, 0.2, 0.3],
                              ch_type='eeg',
                              time_unit='s',
                              size=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
