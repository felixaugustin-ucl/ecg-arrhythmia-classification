#!/usr/bin/env python3
"""Train the 1D ResNet on raw 250 Hz signal.

    python scripts/05_train_cnn.py --epochs 8

Requires scripts/04_signal_pca.py to have written the resampled tensor.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ecg.config import (
    ARTIFACT_DIR,
    RANDOM_SEED,
    TABLE_DIR,
    TEST_SIZE,
    VAL_SIZE,
    ensure_output_dirs,
    load_config,
)
from ecg.data import label_sets
from ecg.evaluation import best_cutoff, multilabel_metrics, per_label_metrics, sweep_cutoffs
from ecg.models.cnn import TrainConfig, predict_proba, seed_torch, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_output_dirs()
    seed_torch(args.seed)
    config = load_config("models/cnn_250hz")

    signal_path = ARTIFACT_DIR / "signals_250hz.npy"
    if not signal_path.exists():
        raise SystemExit(f"{signal_path} not found — run scripts/04_signal_pca.py first.")
    X = np.load(signal_path)

    metadata = pd.DataFrame(
        json.loads(str(np.load(ARTIFACT_DIR / "dataset.npz", allow_pickle=True)["metadata"]))
    )
    from sklearn.preprocessing import MultiLabelBinarizer

    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(label_sets(metadata)).astype(np.float32)
    print(f"Signals {X.shape}, labels {y.shape}")

    # The test split is carved out first and never touched again; validation
    # comes out of the training portion only.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=args.seed, shuffle=True
    )
    print(f"  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    training = config["training"]
    train_config = TrainConfig(
        epochs=args.epochs or training["epochs"],
        batch_size=training["batch_size"],
        eval_batch_size=training["eval_batch_size"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        early_stopping_patience=training["early_stopping_patience"],
        base_channels=config["architecture"]["base_channels"],
        layers=tuple(config["architecture"]["layers"]),
        device=args.device or training["device"],
    )

    model, history = train(X_train, y_train, X_val, y_val, train_config)
    pd.DataFrame(history).to_csv(TABLE_DIR / "cnn_250hz_history.csv", index=False)

    # Cutoff chosen on validation, applied to test — never chosen on test.
    val_proba = predict_proba(model, X_val, train_config.eval_batch_size, train_config.device)
    sweep = sweep_cutoffs(y_val, val_proba)
    sweep.to_csv(TABLE_DIR / "cnn_250hz_cutoff_sweep.csv", index=False)

    threshold = config.get("threshold", {})
    metric = threshold.get("selection_metric", "f1_micro")
    cutoff = threshold.get("optimal_cutoff") or best_cutoff(sweep, metric)
    print(f"  cutoff {cutoff:.2f} (selected on validation {metric})")

    test_proba = predict_proba(model, X_test, train_config.eval_batch_size, train_config.device)
    y_pred = (test_proba >= cutoff).astype(int)

    scores = multilabel_metrics(y_test, y_pred)
    print("\n  held-out test performance:")
    for name, value in scores.items():
        print(f"    {name:<18} {value:.4f}")

    per_label_metrics(y_test, y_pred, binarizer.classes_).to_csv(
        TABLE_DIR / "cnn_250hz_per_label.csv", index=False
    )
    pd.DataFrame([{"model": "cnn_250hz", "cutoff": cutoff, **scores}]).to_csv(
        TABLE_DIR / "cnn_250hz_summary.csv", index=False
    )

    import torch

    torch.save(model.state_dict(), ARTIFACT_DIR / "cnn_250hz.pt")
    print(f"\n  saved model to {ARTIFACT_DIR / 'cnn_250hz.pt'}")


if __name__ == "__main__":
    main()
