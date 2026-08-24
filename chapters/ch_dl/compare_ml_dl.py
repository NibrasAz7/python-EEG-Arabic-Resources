"""Compare deep learning (EEGNet) with classical ML on EEG data.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, trains EEGNet on raw signal and LogisticRegression on band
power features (delta, theta, alpha, beta), and compares accuracy
using 5-fold cross-validation with error bars showing std across folds.

Usage:
    python compare_ml_dl.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

OUTPUT_DIR = Path(__file__).resolve().parent
FS = 250
BANDS = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'), (13, 30, 'beta')]


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_samples=1001, n_classes=2, F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, n_samples // 2), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same'),
            nn.Conv2d(F1 * D, F2, (1, 1)),
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        dummy = torch.zeros(1, 1, n_channels, n_samples)
        out = self._features(dummy)
        self.classify = nn.Linear(out.view(-1).shape[0], n_classes)

    def _features(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


def load_data(subject):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]
    return X, labels


def labels_to_int(labels):
    return np.array([0 if lab == 'left_hand' else 1 for lab in labels])


def train_eegnet(X_train, y_train, n_channels, n_samples, epochs=30, lr=0.001, batch_size=32):
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    n_samples_train = X_train_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples_train, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"  Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / n_batches:.4f}")
    return model


def evaluate_eegnet(model, X_test, y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
    return accuracy_score(y_test, predicted.numpy())


def extract_band_power_features(X):
    n_trials, n_channels, n_samples = X.shape
    features = []
    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            freqs, psd = welch(X[trial, ch, :], fs=FS, nperseg=256)
            for fmin, fmax, name in BANDS:
                mask = (freqs >= fmin) & (freqs <= fmax)
                power = np.trapezoid(psd[mask], freqs[mask])
                trial_features.append(power)
        features.append(trial_features)
    return np.array(features)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    X, labels = load_data(subject=1)
    y = labels_to_int(labels)
    print(f"Data shape: {X.shape}")

    n_channels = X.shape[1]
    n_samples = X.shape[2]

    print("Extracting band power features...")
    features = extract_band_power_features(X)
    print(f"Feature matrix shape: {features.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dl_accs = []
    ml_accs = []

    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        print(f"\nFold {fold}/5")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(" Training EEGNet...")
        torch.manual_seed(42)
        model = train_eegnet(X_train_raw, y_train, n_channels, n_samples, epochs=30)
        dl_acc = evaluate_eegnet(model, X_test_raw, y_test)
        dl_accs.append(dl_acc)
        print(f" DL accuracy: {dl_acc:.4f}")

        X_train_feat = features[train_idx]
        X_test_feat = features[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        ml_acc = accuracy_score(y_test, y_pred)
        ml_accs.append(ml_acc)
        print(f" ML accuracy: {ml_acc:.4f}")

    dl_mean = np.mean(dl_accs)
    dl_std = np.std(dl_accs)
    ml_mean = np.mean(ml_accs)
    ml_std = np.std(ml_accs)

    print(f"\nDL (EEGNet): {dl_mean:.4f} +/- {dl_std:.4f}")
    print(f"ML (LR + power): {ml_mean:.4f} +/- {ml_std:.4f}")

    methods = ['DL (EEGNet)', 'ML (LR + power)']
    means = [dl_mean, ml_mean]
    stds = [dl_std, ml_std]
    colors = ['steelblue', 'coral']

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(methods, means, yerr=stds, color=colors, edgecolor='black', width=0.5, capsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Deep Learning vs Classical ML (5-fold CV)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02, f'{mean:.4f}\n+/- {std:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'compare_ml_dl_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
