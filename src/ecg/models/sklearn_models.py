"""The scikit-learn and XGBoost models, as registry entries.

All four are one-vs-rest over the multi-label target. The linear models are
scaled inside the pipeline so that cross-validation never sees statistics from
its own held-out fold; the tree models are left unscaled, which is what they
want anyway.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg.config import RANDOM_SEED
from ecg.models.registry import ModelSpec, register

#: SGD losses that expose ``predict_proba`` — required for cutoff tuning.
PROBABILISTIC_LOSSES = ("log_loss", "modified_huber")


def build_sgd(config: dict[str, Any]):
    """One-vs-rest SGD with standardisation folded into the pipeline."""
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            SGDClassifier(
                loss=config["loss"],
                alpha=float(config["alpha"]),
                penalty=config["penalty"],
                learning_rate=config["learning_rate"],
                eta0=0.01,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-3,
                random_state=RANDOM_SEED,
            )
        ),
    )


def build_xgb(config: dict[str, Any]):
    """One-vs-rest gradient-boosted trees."""
    from xgboost import XGBClassifier

    return OneVsRestClassifier(
        XGBClassifier(
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            n_estimators=int(config["n_estimators"]),
            subsample=float(config.get("subsample", 1.0)),
            colsample_bytree=float(config.get("colsample_bytree", 1.0)),
            min_child_weight=float(config.get("min_child_weight", 1.0)),
            gamma=float(config.get("gamma", 0.0)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            reg_alpha=float(config.get("reg_alpha", 0.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
    )


def _sgd_space() -> dict[str, Any]:
    from ray import tune

    return {
        "alpha": tune.loguniform(1e-6, 1e-1),
        # Restricted to probabilistic losses: hinge has no predict_proba, and
        # the notebook's search wasted trials on configs it then had to discard.
        "loss": tune.choice(list(PROBABILISTIC_LOSSES)),
        "penalty": tune.choice(["l1", "l2", "elasticnet"]),
        "learning_rate": tune.choice(["optimal", "adaptive", "constant", "invscaling"]),
    }


def _xgb_space() -> dict[str, Any]:
    from ray import tune

    return {
        "max_depth": tune.randint(2, 9),
        "learning_rate": tune.loguniform(1e-3, 3e-1),
        "n_estimators": tune.randint(50, 400),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.loguniform(1e-1, 1e1),
        "gamma": tune.uniform(0.0, 5.0),
        "reg_lambda": tune.loguniform(1e-2, 1e2),
        "reg_alpha": tune.loguniform(1e-8, 1e1),
    }


SGD_PARAMS = ("alpha", "loss", "penalty", "learning_rate")
XGB_PARAMS = (
    "max_depth", "learning_rate", "n_estimators", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "reg_lambda", "reg_alpha",
)

register(ModelSpec(
    name="sgd_18f",
    description="One-vs-rest SGD on 18 handcrafted ECG features",
    feature_source="metrics_18f",
    build=build_sgd,
    search_space=_sgd_space,
    param_names=SGD_PARAMS,
))

register(ModelSpec(
    name="sgd_13c",
    description="One-vs-rest SGD on 13 PCA components of the 18 features",
    feature_source="metrics_pca_13c",
    build=build_sgd,
    search_space=_sgd_space,
    param_names=SGD_PARAMS,
))

register(ModelSpec(
    name="xgb_18f",
    description="One-vs-rest XGBoost on 18 handcrafted ECG features (unscaled)",
    feature_source="metrics_18f",
    build=build_xgb,
    search_space=_xgb_space,
    param_names=XGB_PARAMS,
))

register(ModelSpec(
    name="xgb_100f",
    description="One-vs-rest XGBoost on 100 Incremental-PCA components of the raw signal",
    feature_source="signal_pca_100",
    build=build_xgb,
    search_space=_xgb_space,
    param_names=XGB_PARAMS,
))
