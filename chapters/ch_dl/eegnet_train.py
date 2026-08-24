"""Train EEGNet on MOABB motor imagery data.

Loads BNCI2014-001 subject 1, filters to left_hand and right_hand
classes, builds the EEGNet architecture (Lawhern et al. 2018) in
PyTorch, trains for 50 epochs with Adam, and plots the training loss
curve and the confusion matrix on the held-out test set.

Usage:
    python eegnet_train.py
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
from sklearn.metrics import confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent
CLASSES = ['left_hand', 'right_hand']


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


def train_model(model, X_train, y_train, epochs=50, lr=0.001, batch_size=32):
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
    accuracy = np.mean(predicted == y_test.numpy())
    cm = confusion_matrix(y_test.numpy(), predicted, labels=[0, 1])
    return accuracy, cm


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, labels = load_data(subject=1)
    print(f"Data shape: {X.shape}, labels: {len(labels)}")

    y = np.array([0 if lab == 'left_hand' else 1 for lab in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    n_channels = X.shape[1]
    n_samples = X.shape[2]
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=2).to(device)
    print(model)

    losses = train_model(model, X_train_t, y_train_t, epochs=50, lr=0.001)

    accuracy, cm = evaluate_model(model, X_test_t, y_test_t)
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Confusion matrix:\n{cm}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(range(1, len(losses) + 1), losses, 'o-', color='steelblue', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('EEGNet Training Loss Curve')
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(CLASSES, rotation=45)
    axes[1].set_yticklabels(CLASSES)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    axes[1].set_title(f'Confusion Matrix (accuracy={accuracy:.4f})')
    plt.colorbar(im, ax=axes[1])

    plt.suptitle('EEGNet Training on BNCI2014-001 (Subject 1)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'eegnet_train_result.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
