#!/usr/bin/env python3
"""Train and evaluate one model. The same path serves all five.

    python scripts/03_train.py --model sgd_18f
    python scripts/03_train.py --model xgb_100f --retune --trials 50
    python scripts/03_train.py --model cnn_250hz

By default the frozen hyperparameters in ``configs/models/<model>.yaml`` are
used, which reproduces the reported results in minutes rather than hours.
``--retune`` re-runs the Ray search and prints the config it found, so the YAML
can be updated deliberately rather than by pasting into a notebook cell.
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
    ensure_output_dirs,
    load_config,
    seed_everything,
)
from ecg.evaluation import best_cutoff, multilabel_metrics, per_label_metrics, sweep_from_cv
from ecg.models import available_models, get_model_spec
from ecg.preprocessing import prepare


def load_features() -> pd.DataFrame:
    path = ARTIFACT_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/02_extract_features.py first.")
    return pd.read_parquet(path)


def load_label_sets() -> pd.Series:
    path = ARTIFACT_DIR / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/01_build_dataset.py first.")
    metadata = pd.DataFrame(json.loads(str(np.load(path, allow_pickle=True)["metadata"])))
    from ecg.data import label_sets

    return label_sets(metadata)


def build_feature_matrix(spec, features: pd.DataFrame, config: dict):
    """Return the matrix this model's ``feature_source`` calls for."""
    if spec.feature_source == "metrics_18f":
        return features

    if spec.feature_source == "metrics_pca_13c":
        from ecg.decomposition import fit_feature_pca

        n_components = config.get("pca", {}).get("n_components", 13)
        result = fit_feature_pca(features.to_numpy(), n_components=n_components)
        projected = result.model.transform(result.scaler.transform(features.to_numpy()))
        print(
            f"  PCA {features.shape[1]} -> {n_components} components, "
            f"cumulative variance {result.cumulative_explained_variance[-1]:.4f}"
        )
        return pd.DataFrame(projected, columns=[f"pc{i + 1}" for i in range(n_components)])

    raise NotImplementedError(
        f"feature_source {spec.feature_source!r} needs the raw signal tensor; "
        "run scripts/04_signal_pca.py to build it first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=available_models() + ["cnn_250hz"])
    parser.add_argument("--retune", action="store_true", help="re-run the Ray search")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_output_dirs()
    seed_everything(args.seed)

    if args.model == "cnn_250hz":
        raise SystemExit(
            "cnn_250hz trains on the raw signal tensor: use scripts/05_train_cnn.py"
        )

    spec = get_model_spec(args.model)
    config = load_config(f"models/{args.model}")
    print(f"{spec.name}: {spec.description}")

    features = load_features()
    data = prepare(features, load_label_sets())

    # Build the model's feature representation first, then split it — so the
    # split indices always refer to the matrix actually being trained on.
    X = build_feature_matrix(spec, data.X, config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.y, test_size=TEST_SIZE, random_state=args.seed, shuffle=True
    )
    print(f"  train {X_train.shape}, test {X_test.shape}, labels {data.y.shape[1]}")

    if args.retune:
        from ecg.tuning import run_search

        tuning = load_config(f"models/{args.model}").get("tuning", {})
        result = run_search(
            spec,
            X_train,
            y_train,
            n_trials=args.trials or tuning.get("n_trials", 30),
            cv_folds=tuning.get("cv_folds", 3),
            seed=args.seed,
        )
        model_config = result.best_config
        result.results.to_csv(TABLE_DIR / f"{spec.name}_tuning_trials.csv", index=False)
        print(f"\n  best config (trial {result.best_trial_id}):")
        print(json.dumps({k: str(v) for k, v in model_config.items()}, indent=4))
        print(f"  update configs/models/{args.model}.yaml with these values to freeze them")
    else:
        model_config = {k: v for k, v in config["best_config"].items() if k != "trial_id"}
        print(f"  using frozen config from configs/models/{args.model}.yaml")

    model = spec.build(model_config)

    threshold_cfg = config.get("threshold", {})
    print("\n  sweeping probability cutoffs on training folds")
    sweep, _ = sweep_from_cv(model, X_train, y_train, cv=threshold_cfg.get("cv_folds", 5))
    sweep.to_csv(TABLE_DIR / f"{spec.name}_cutoff_sweep.csv", index=False)

    metric = threshold_cfg.get("selection_metric", "f1_micro")
    cutoff = threshold_cfg.get("optimal_cutoff") or best_cutoff(sweep, metric)
    print(f"  cutoff {cutoff:.2f} (selected on {metric})")

    print("\n  fitting on the full training split")
    model.fit(
        X_train.to_numpy(dtype=np.float32) if hasattr(X_train, "to_numpy") else X_train, y_train
    )
    y_proba = model.predict_proba(
        X_test.to_numpy(dtype=np.float32) if hasattr(X_test, "to_numpy") else X_test
    )
    y_pred = (np.asarray(y_proba) >= cutoff).astype(int)

    scores = multilabel_metrics(y_test, y_pred)
    print("\n  held-out test performance:")
    for name, value in scores.items():
        print(f"    {name:<18} {value:.4f}")

    per_label = per_label_metrics(y_test, y_pred, data.classes)
    per_label.to_csv(TABLE_DIR / f"{spec.name}_per_label.csv", index=False)
    pd.DataFrame([{"model": spec.name, "cutoff": cutoff, **scores}]).to_csv(
        TABLE_DIR / f"{spec.name}_summary.csv", index=False
    )
    print(f"\n  wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
