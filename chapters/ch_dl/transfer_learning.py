"""Demonstrate transfer learning between subjects with EEGNet.

Loads BNCI2014-001 subjects 1 and 2, filters to left_hand and
right_hand classes, trains EEGNet on subject 1, evaluates zero-shot
transfer on subject 2, fine-tunes the pre-trained model on subject 2,
and compares against a model trained from scratch on subject 2.

Usage:
    python transfer_learning.py
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


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_samples=1001, n_classes=2, F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D),
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


def train_model(model, X_train, y_train, epochs=30, lr=0.001, batch_size=32):
    device = next(model.parameters()).device
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
            batch_x = X_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
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
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    accuracy = accuracy_score(y_test.numpy(), predicted)
    return accuracy


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X1, labels1 = load_data(subject=1)
    y1 = labels_to_int(labels1)
    print(f"Subject 1 data shape: {X1.shape}")

    X2, labels2 = load_data(subject=2)
    y2 = labels_to_int(labels2)
    print(f"Subject 2 data shape: {X2.shape}")

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )

    X1_train_t = torch.tensor(X1_train, dtype=torch.float32).unsqueeze(1)
    X1_test_t = torch.tensor(X1_test, dtype=torch.float32).unsqueeze(1)
    y1_train_t = torch.tensor(y1_train, dtype=torch.long)
    y1_test_t = torch.tensor(y1_test, dtype=torch.long)

    X2_train_t = torch.tensor(X2_train, dtype=torch.float32).unsqueeze(1)
    X2_test_t = torch.tensor(X2_test, dtype=torch.float32).unsqueeze(1)
    y2_train_t = torch.tensor(y2_train, dtype=torch.long)
    y2_test_t = torch.tensor(y2_test, dtype=torch.long)

    n_channels = X1.shape[1]
    n_samples = X1.shape[2]

    print("Training EEGNet on subject 1 (30 epochs)...")
    model_s1 = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2).to(device)
    train_model(model_s1, X1_train_t, y1_train_t, epochs=30, lr=0.001)
    acc_s1 = evaluate_model(model_s1, X1_test_t, y1_test_t)
    print(f"Subject 1 test accuracy: {acc_s1:.4f}")

    print("Evaluating zero-shot transfer (S1 model on S2 test)...")
    acc_zeroshot = evaluate_model(model_s1, X2_test_t, y2_test_t)
    print(f"Zero-shot S1->S2 accuracy: {acc_zeroshot:.4f}")

    print("Fine-tuning pre-trained model on subject 2 (10 epochs, lr=0.0001)...")
    model_finetune = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2).to(device)
    model_finetune.load_state_dict(model_s1.state_dict())
    train_model(model_finetune, X2_train_t, y2_train_t, epochs=10, lr=0.0001)
    acc_finetune = evaluate_model(model_finetune, X2_test_t, y2_test_t)
    print(f"Fine-tuned S1->S2 accuracy: {acc_finetune:.4f}")

    torch.manual_seed(42)
    np.random.seed(42)
    print("Training fresh model on subject 2 from scratch (30 epochs)...")
    model_scratch = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2).to(device)
    train_model(model_scratch, X2_train_t, y2_train_t, epochs=30, lr=0.001)
    acc_scratch = evaluate_model(model_scratch, X2_test_t, y2_test_t)
    print(f"S2 from scratch accuracy: {acc_scratch:.4f}")

    scenarios = ['Train S1,\nTest S2\n(zero-shot)', 'Fine-tune\nS1->S2\n(transfer)', 'Train S2\nfrom scratch\n(baseline)']
    accuracies = [acc_zeroshot, acc_finetune, acc_scratch]
    colors = ['steelblue', 'coral', 'seagreen']

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(scenarios, accuracies, color=colors, edgecolor='black', width=0.5)
    ax.set_ylabel('Test Accuracy on Subject 2')
    ax.set_title('Transfer Learning: EEGNet Between Subjects')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transfer_learning_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
