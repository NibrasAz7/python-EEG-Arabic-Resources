"""KNN-based artifact detection in EEG signals.

Extracts statistical features (variance, kurtosis, max amplitude)
from sliding windows of the P4 channel, labels windows as clean
or artifact based on amplitude threshold, trains a KNN classifier,
and visualizes detected artifact regions.

Usage:
    python knn_detect.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score

from utils.eeg_loader import load_local_eeg

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local"
FS = 200
WINDOW_SIZE = 200


def main() -> None:
    timestamps, eeg_data, ch_names = load_local_eeg(
        data_dir=DATA_DIR, subject=7, experiment=1, session=2
    )
    channel_data = eeg_data[:, 0]

    n_windows = len(channel_data) // WINDOW_SIZE
    windows = channel_data[:n_windows * WINDOW_SIZE].reshape(n_windows, WINDOW_SIZE)

    features = np.column_stack([
        np.var(windows, axis=1),
        kurtosis(windows, axis=1),
        np.max(np.abs(windows), axis=1),
    ])

    threshold = 3 * np.std(channel_data)
    labels = (np.max(np.abs(windows), axis=1) > threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    # Use cross-validated predictions for all windows to avoid data leakage
    # (predicting on training data would inflate the visualization)
    predictions = cross_val_predict(
        KNeighborsClassifier(n_neighbors=5), features, labels, cv=5
    )
    accuracy = accuracy_score(y_test, knn.predict(X_test))

    time_sec = np.arange(len(channel_data)) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time_sec, channel_data, linewidth=0.5, color='blue')
    for i in range(n_windows):
        if labels[i] == 1:
            t_start = i * WINDOW_SIZE / FS
            t_end = (i + 1) * WINDOW_SIZE / FS
            axes[0].axvspan(t_start, t_end, alpha=0.3, color='red')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title(f'Ground truth artifacts (threshold={threshold:.0f} uV)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_sec, channel_data, linewidth=0.5, color='blue')
    for i in range(n_windows):
        if predictions[i] == 1:
            t_start = i * WINDOW_SIZE / FS
            t_end = (i + 1) * WINDOW_SIZE / FS
            axes[1].axvspan(t_start, t_end, alpha=0.3, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')
    axes[1].set_title(f'KNN predictions (accuracy={accuracy:.1%})')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('KNN Artifact Detection - Channel P4', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(__file__).resolve().parent / 'knn_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
