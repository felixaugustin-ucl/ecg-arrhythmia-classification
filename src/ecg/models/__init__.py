"""Model definitions and the registry that lets scripts stay model-agnostic."""

from ecg.models import sklearn_models  # noqa: F401  (registers SGD/XGB specs)
from ecg.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    available_models,
    get_model_spec,
    register,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "available_models",
    "get_model_spec",
    "register",
]
