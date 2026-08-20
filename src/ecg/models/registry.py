"""A registry mapping model names to how they are built, tuned and fed.

The notebook expressed this with variable-name prefixes: every SGD_18F object
was called ``SGD_18F_something``. A registry does the same job, but the name
becomes data — so ``scripts/03_train.py --model xgb_100f`` works without a
single model-specific branch anywhere in the script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: Which feature representation a model consumes.
FEATURE_SOURCES = ("metrics_18f", "metrics_pca_13c", "signal_pca_100", "raw_signal_250hz")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the shared training path needs to know about one model.

    Attributes:
        name: registry key, e.g. ``"sgd_18f"``.
        description: one line, used in ``--help`` and report headings.
        feature_source: which matrix to train on; see :data:`FEATURE_SOURCES`.
        build: ``config -> estimator``, where config comes from tuning or YAML.
        search_space: ``() -> dict`` of Ray Tune distributions. Deferred so
            importing the registry never requires Ray to be installed.
        supports_proba: whether cutoff sweeping applies.
        tune_metric: the metric the search maximises.
    """

    name: str
    description: str
    feature_source: str
    build: Callable[[dict[str, Any]], Any]
    search_space: Callable[[], dict[str, Any]] | None = None
    supports_proba: bool = True
    tune_metric: str = "f1_macro"
    param_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.feature_source not in FEATURE_SOURCES:
            raise ValueError(
                f"{self.name}: feature_source {self.feature_source!r} not in {FEATURE_SOURCES}"
            )


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Add a spec to the registry, rejecting duplicate names."""
    if spec.name in MODEL_REGISTRY:
        raise ValueError(f"Model {spec.name!r} is already registered.")
    MODEL_REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(available_models())}")
    return MODEL_REGISTRY[name]


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
