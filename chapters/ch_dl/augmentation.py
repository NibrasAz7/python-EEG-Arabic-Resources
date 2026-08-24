"""Demonstrate data augmentation techniques for EEG signals.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, applies three augmentation techniques (time shift, channel
dropout, Gaussian noise) to single-trial examples, and compares
EEGNet accuracy trained with and without augmentation.

Usage:
    python augmentation.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTPUT_DIR = Path(__file__).resolve().parent
CLASSES = ['left_hand', 'right_hand']


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


def time_shift(signal, max_shift=50):
    shifted = np.zeros_like(signal)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return signal.copy()
    if shift > 0:
        shifted[:, shift:] = signal[:, :-shift]
    else:
        shifted[:, :shift] = signal[:, -shift:]
    return shifted


def channel_dropout(signal, p=0.2):
    dropped = signal.copy()
    n_channels = signal.shape[0]
    drop_mask = np.random.rand(n_channels) < p
    dropped[drop_mask] = 0.0
    return dropped


def gaussian_noise(signal, sigma=0.1):
    noisy = signal + np.random.normal(0, sigma, signal.shape)
    return noisy


def load_data(subject):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    mask = (labels == 'left_hand') | (labels == 'right_hand')
    X = X[mask]
    labels = labels[mask]
    return X, labels


def augment_batch(X_batch):
    augmented = np.zeros_like(X_batch)
    for i in range(X_batch.shape[0]):
        choice = np.random.randint(0, 3)
        if choice == 0:
            augmented[i] = time_shift(X_batch[i])
        elif choice == 1:
            augmented[i] = channel_dropout(X_batch[i])
        else:
            augmented[i] = gaussian_noise(X_batch[i])
    return augmented


def train_model(model, X_train, y_train, epochs=30, lr=0.001, batch_size=32, use_augmentation=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n_samples = X_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = X_train[idx].numpy()
            batch_y = y_train[idx]
            if use_augmentation:
                batch_x = augment_batch(batch_x)
            batch_x_t = torch.tensor(batch_x, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x_t)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
    return losses


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.numpy()
    accuracy = accuracy_score(y_test.numpy(), predicted)
    return accuracy


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    X, labels = load_data(subject=1)
    print(f"Data shape: {X.shape}, labels: {len(labels)}")

    y = np.array([0 if lab == 'left_hand' else 1 for lab in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    example = X[0]
    np.random.seed(0)
    ex_shift = time_shift(example, max_shift=50)
    ex_dropout = channel_dropout(example, p=0.2)
    ex_noise = gaussian_noise(example, sigma=0.1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    n_channels = X.shape[1]
    n_samples = X.shape[2]

    print("Training without augmentation...")
    model_plain = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2)
    train_model(model_plain, X_train_t, y_train_t, epochs=30, lr=0.001, use_augmentation=False)
    acc_plain = evaluate_model(model_plain, X_test_t, y_test_t)
    print(f"Accuracy without augmentation: {acc_plain:.4f}")

    torch.manual_seed(42)
    np.random.seed(42)
    print("Training with augmentation...")
    model_aug = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2)
    train_model(model_aug, X_train_t, y_train_t, epochs=30, lr=0.001, use_augmentation=True)
    acc_aug = evaluate_model(model_aug, X_test_t, y_test_t)
    print(f"Accuracy with augmentation: {acc_aug:.4f}")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(example[0], label='Original', alpha=0.7)
    ax0.plot(ex_shift[0], label='Shifted', alpha=0.7)
    ax0.set_title('Time Shift')
    ax0.set_xlabel('Samples')
    ax0.set_ylabel('Amplitude')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(example[0], label='Original', alpha=0.7)
    ax1.plot(ex_dropout[0], label='Dropped', alpha=0.7)
    ax1.set_title('Channel Dropout')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(example[0], label='Original', alpha=0.7)
    ax2.plot(ex_noise[0], label='Noisy', alpha=0.7)
    ax2.set_title('Gaussian Noise')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Amplitude')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, :])
    methods = ['No Augmentation', 'With Augmentation']
    accuracies = [acc_plain, acc_aug]
    bars = ax_bar.bar(methods, accuracies, color=['steelblue', 'coral'], edgecolor='black', width=0.5)
    ax_bar.set_ylabel('Test Accuracy')
    ax_bar.set_title('EEGNet Accuracy: Augmentation Comparison')
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontsize=12)

    plt.suptitle('Data Augmentation for EEG (BNCI2014-001, Subject 1)', fontsize=14)
    plt.savefig(OUTPUT_DIR / 'augmentation_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
