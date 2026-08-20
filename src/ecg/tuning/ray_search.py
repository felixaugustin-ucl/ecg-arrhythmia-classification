"""One Ray Tune harness, parameterised by :class:`~ecg.models.registry.ModelSpec`.

The notebook repeated this block four times, once per model, with the model
name spliced into every identifier. The only genuine differences were the
estimator, the search space and the parameter names — all of which the spec
already carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from ecg.config import RANDOM_SEED
from ecg.models.registry import ModelSpec

SCORING = {
    "precision_micro": "precision_micro",
    "precision_macro": "precision_macro",
    "recall_micro": "recall_micro",
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
}

REPORTED_METRICS = tuple(SCORING)


@dataclass
class TuningResult:
    """Outcome of one search: the full trial table and the selected config."""

    results: pd.DataFrame
    best_config: dict[str, Any]
    best_trial_id: str
    metric: str
    n_trials: int
    cv_folds: int

    def best_by_metric(self, metrics: tuple[str, ...] = REPORTED_METRICS) -> pd.DataFrame:
        """The top trial under each metric — shows how much the choice matters."""
        rows = []
        for metric in metrics:
            if metric in self.results.columns and self.results[metric].notna().any():
                row = self.results.loc[self.results[metric].idxmax()]
                rows.append({"selection_metric": metric, **row.to_dict()})
        return pd.DataFrame(rows)


def normalise_results(analysis, param_names: tuple[str, ...]) -> pd.DataFrame:
    """Flatten a Ray ``ExperimentAnalysis`` into a tidy trial table.

    Ray reports metrics under ``metric``, ``last_result/metric`` or
    ``last_result.metric`` depending on version and scheduler; this checks all
    three rather than assuming one.
    """
    df = analysis.dataframe().copy()
    df = df.rename(columns={f"config/{name}": name for name in param_names})

    for metric in REPORTED_METRICS:
        if metric in df.columns:
            continue
        for alternative in (f"last_result/{metric}", f"last_result.{metric}"):
            if alternative in df.columns:
                df[metric] = df[alternative]
                break

    keep = [
        column
        for column in ("trial_id", *param_names, *REPORTED_METRICS,
                       "training_iteration", "time_total_s", "timestamp")
        if column in df.columns
    ]
    return df[keep].copy()


def _make_objective(spec: ModelSpec, cv_folds: int):
    """Build the trainable Ray executes per trial."""
    from ray import tune

    def objective(config, X_train_arr, y_train_arr):
        scores = cross_validate(
            spec.build(config),
            X_train_arr,
            y_train_arr,
            cv=cv_folds,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        tune.report(
            {metric: float(np.mean(scores[f"test_{metric}"])) for metric in REPORTED_METRICS}
        )

    return objective


def _search_algorithm(seed: int):
    """Prefer Optuna's TPE; fall back to HyperOpt if it is unavailable."""
    try:
        from ray.tune.search.optuna import OptunaSearch

        return OptunaSearch(seed=seed), "optuna"
    except ImportError:
        from ray.tune.search.hyperopt import HyperOptSearch

        return HyperOptSearch(random_state_seed=seed), "hyperopt"


def run_search(
    spec: ModelSpec,
    X_train,
    y_train,
    n_trials: int = 30,
    cv_folds: int = 3,
    seed: int = RANDOM_SEED,
    verbose: int = 1,
) -> TuningResult:
    """Run hyperparameter search for ``spec`` on the training split only."""
    import random

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    if spec.search_space is None:
        raise ValueError(f"{spec.name} has no search space; it is not tunable via Ray.")

    X_arr = (
        X_train.to_numpy(dtype=np.float32)
        if hasattr(X_train, "to_numpy")
        else np.asarray(X_train, dtype=np.float32)
    )
    y_arr = np.asarray(y_train)

    np.random.seed(seed)
    random.seed(seed)
    search_alg, backend = _search_algorithm(seed)
    print(f"{spec.name}: search backend {backend}, {n_trials} trials, {cv_folds}-fold CV")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    try:
        analysis = tune.run(
            tune.with_parameters(
                _make_objective(spec, cv_folds), X_train_arr=X_arr, y_train_arr=y_arr
            ),
            config=spec.search_space(),
            num_samples=n_trials,
            metric=spec.tune_metric,
            mode="max",
            resources_per_trial={"cpu": 1},
            search_alg=search_alg,
            # max_t=1: each trial is a full CV run reporting once, so there is
            # nothing to halve early. ASHA is kept for its pruning bookkeeping.
            scheduler=ASHAScheduler(max_t=1, grace_period=1, reduction_factor=2),
            verbose=verbose,
        )
    finally:
        ray.shutdown()

    results = normalise_results(analysis, spec.param_names)
    sort_columns = [c for c in (spec.tune_metric, "f1_micro") if c in results.columns]
    results = results.sort_values(
        sort_columns, ascending=[False] * len(sort_columns), na_position="last"
    ).reset_index(drop=True)

    if results.empty:
        raise RuntimeError(f"{spec.name}: search produced no completed trials.")

    best = results.iloc[0]
    best_config = {name: best[name] for name in spec.param_names if name in results.columns}

    return TuningResult(
        results=results,
        best_config=best_config,
        best_trial_id=str(best.get("trial_id", "")),
        metric=spec.tune_metric,
        n_trials=n_trials,
        cv_folds=cv_folds,
    )
