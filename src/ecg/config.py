"""Paths, constants and configuration loading.

Every magic number that the original notebook scattered across cells lives
here or in ``configs/``, so a run can be described by its config alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Overridable so the 5.3 GB download can live on an external volume.
DATA_ROOT = Path(os.environ.get("ECG_DATA_ROOT", PROJECT_ROOT / "ecg_data"))

RANDOM_SEED = 42

#: Native sampling rate of the PhysioNet recordings.
SOURCE_FS_HZ = 500
#: Rate the raw-signal models (Incremental PCA, CNN) are trained at.
TARGET_FS_HZ = 250

#: Lead index used for handcrafted feature extraction (0-based; lead II).
FEATURE_LEAD_INDEX = 1

TEST_SIZE = 0.20
VAL_SIZE = 0.20


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem locations for one run."""

    data_root: Path = DATA_ROOT
    records: Path = field(init=False)
    condition_names: Path = field(init=False)
    remaining_codes: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", self.data_root / "WFDBRecords")
        object.__setattr__(
            self, "condition_names", self.data_root / "ConditionNames_SNOMED-CT.csv"
        )
        object.__setattr__(
            self, "remaining_codes", self.data_root / "Remaining_DX_Codes_SNOMED_Labels.csv"
        )

    def require(self) -> Paths:
        """Fail early with an actionable message if the dataset is absent."""
        if not self.records.is_dir():
            raise FileNotFoundError(
                f"No WFDB records under {self.records}.\n"
                "Download the dataset first (see README, 'Getting the data'), or point "
                "ECG_DATA_ROOT at an existing copy."
            )
        return self


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by name, e.g. ``load_config('models/xgb_18f')``."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(
            p.relative_to(CONFIG_DIR).with_suffix("").as_posix()
            for p in CONFIG_DIR.rglob("*.yaml")
        )
        raise FileNotFoundError(f"No config {name!r}. Available: {', '.join(available)}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def ensure_output_dirs() -> None:
    for directory in (FIGURE_DIR, TABLE_DIR, ARTIFACT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and (if installed) torch in one call."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
