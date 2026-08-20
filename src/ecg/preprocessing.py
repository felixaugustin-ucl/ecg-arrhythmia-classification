"""Turn the raw feature table into model-ready matrices.

The original notebook prepared this once and reused it across SGD_18F,
XGB_18F and (after PCA) SGD_13C. That sharing is preserved here, but made
explicit: one function, one return value, no cross-cell globals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from ecg.config import RANDOM_SEED, TEST_SIZE

#: Dropped before modelling: both are RR-derived complexity measures that are
#: undefined for short records, so they carry the most missingness of the 22.
DEFAULT_DROP_COLUMNS = ("higuchi_fractal_dimension", "dfa_scaling_exponent")


@dataclass
class ModelingData:
    """A prepared multi-label dataset with its fitted transformers."""

    X: pd.DataFrame
    y: np.ndarray
    label_binarizer: MultiLabelBinarizer
    imputer: SimpleImputer

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def classes(self) -> np.ndarray:
        return self.label_binarizer.classes_

    def split(self, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
        """Stratification is not available for multi-label targets, so this is
        a plain shuffled split with a fixed seed."""
        return train_test_split(
            self.X, self.y, test_size=test_size, random_state=seed, shuffle=True
        )


def prepare(
    metrics: pd.DataFrame,
    label_sets: pd.Series,
    drop_columns: tuple[str, ...] = DEFAULT_DROP_COLUMNS,
) -> ModelingData:
    """Drop unusable columns, median-impute, and binarise the label sets.

    Infinities (which ``sample_entropy`` can legitimately produce) are treated
    as missing rather than clipped, so they are imputed like any other gap.
    """
    X = metrics.drop(columns=list(drop_columns), errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Columns that are entirely missing cannot be imputed; drop them and say so.
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    ).astype(np.float32).reset_index(drop=True)

    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(label_sets.reset_index(drop=True))

    return ModelingData(X=X, y=y, label_binarizer=binarizer, imputer=imputer)
