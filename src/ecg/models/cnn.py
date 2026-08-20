"""1D ResNet over raw 250 Hz multi-lead ECG.

Defined once. The notebook declared ``BasicBlock1D`` and ``ResNet1D`` twice,
in two cells, and whichever ran last silently won — a genuine hazard, since
the two definitions were not identical.

Kept in its own module (not imported by ``ecg.models.__init__``) so that the
rest of the package works without torch installed.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ecg.config import RANDOM_SEED


class BasicBlock1D(nn.Module):
    """Residual block: two convolutions plus an identity or projection skip."""

    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Project the skip path only when shape actually changes.
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """Compact ResNet for multi-label classification of multi-lead ECG.

    The wide stride-2 stem cuts sequence length early, which is what makes a
    2500-sample, 12-lead input tractable on CPU.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int] = (2, 2, 2),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_channels * 4, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        modules = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        modules += [BasicBlock1D(out_channels, out_channels, stride=1) for _ in range(1, blocks)]
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.head(self.pool(x).squeeze(-1))


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train`, mirroring ``configs/models/cnn_250hz.yaml``."""

    epochs: int = 8
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    base_channels: int = 32
    layers: tuple[int, int, int] = (2, 2, 2)
    device: str = "cpu"
    log_every: int = 100


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_proba(
    model: nn.Module, X: np.ndarray, batch_size: int = 64, device: str = "cpu"
) -> np.ndarray:
    """Sigmoid probabilities for a multi-label head, in eval mode."""
    model.eval()
    outputs = []
    for (xb,) in DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))),
        batch_size=batch_size,
    ):
        logits = model(xb.to(device, dtype=torch.float32))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train with early stopping on validation loss.

    Returns the model restored to its best-validation weights, plus the
    per-epoch history — so the caller reports the best epoch, not the last.
    """
    config = config or TrainConfig()
    device = torch.device(config.device)

    model = ResNet1D(
        in_channels=X_train.shape[1],
        num_classes=y_train.shape[1],
        layers=tuple(config.layers),
        base_channels=config.base_channels,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_loader = _loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, config.eval_batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, dtype=torch.float32)
                yb = yb.to(device, dtype=torch.float32)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
                val_seen += xb.size(0)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss / max(val_seen, 1),
            "elapsed_s": time.time() - started,
        }
        history.append(epoch_stats)
        if verbose:
            print(
                f"epoch {epoch}/{config.epochs}  "
                f"train {epoch_stats['train_loss']:.4f}  val {epoch_stats['val_loss']:.4f}"
            )

        if epoch_stats["val_loss"] < best_val_loss:
            best_val_loss = epoch_stats["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def seed_torch(seed: int = RANDOM_SEED) -> None:
    """Deterministic-as-practical seeding; some CUDA kernels remain nondeterministic."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
