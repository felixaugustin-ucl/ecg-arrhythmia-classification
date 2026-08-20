"""Read WFDB ``.mat``/``.hea`` record pairs into arrays and a metadata table."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class EcgDataset:
    """A padded signal tensor plus its aligned metadata.

    Attributes:
        signals: ``(record, lead, sample)``, NaN-padded to a common shape.
        metadata: one row per record, index-aligned with ``signals``.
    """

    signals: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.signals) != len(self.metadata):
            raise ValueError(
                f"signals ({len(self.signals)}) and metadata ({len(self.metadata)}) are misaligned"
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def lead(self, record_idx: int, lead_idx: int) -> np.ndarray:
        """Return one lead, trimmed to that record's true length (no padding)."""
        n_samples = int(self.metadata.iloc[record_idx]["n_samples"])
        return self.signals[record_idx, lead_idx, :n_samples]


def extract_signal_matrix(mat_dict: dict) -> np.ndarray:
    """Pull the signal matrix out of a loaded ``.mat`` file.

    Prefers the standard WFDB ``val`` key, else the first numeric matrix.
    """
    val = mat_dict.get("val")
    if isinstance(val, np.ndarray):
        return np.asarray(val, dtype=np.float32)
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return np.asarray(value, dtype=np.float32)
    raise ValueError("No numeric signal matrix found in .mat file.")


def read_header_fields(hea_path: Path, wanted: Sequence[str]) -> dict[str, str]:
    """Parse selected ``#Field: value`` comment lines from a WFDB header."""
    fields = {field: "" for field in wanted}
    if not hea_path.exists():
        return fields
    with hea_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for field in wanted:
                if line.startswith(f"#{field}:"):
                    fields[field] = line.split(":", 1)[1].strip()
    return fields


def extract_dx_codes(hea_path: Path) -> list[str]:
    """Diagnosis codes live in the ``#Dx`` field as a comma-separated list."""
    raw = read_header_fields(hea_path, ["Dx"]).get("Dx", "")
    return [code.strip() for code in raw.split(",") if code.strip()]


def _pad_to_tensor(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Stack variable-length records into one NaN-padded tensor."""
    max_leads = max(arr.shape[0] for arr in arrays)
    max_samples = max(arr.shape[1] for arr in arrays)
    tensor = np.full((len(arrays), max_leads, max_samples), np.nan, dtype=np.float32)
    for i, arr in enumerate(arrays):
        leads, samples = arr.shape
        tensor[i, :leads, :samples] = arr
    return tensor


def iter_record_paths(records_root: Path) -> Iterable[Path]:
    yield from sorted(records_root.rglob("*.mat"))


def load_dataset(records_root: Path, limit: int | None = None) -> EcgDataset:
    """Load every record under ``records_root`` into an :class:`EcgDataset`.

    Args:
        records_root: directory containing the WFDB tree.
        limit: read at most this many records — useful for smoke tests, since
            the full dataset is ~45k records and takes minutes to load.
    """
    mat_files = list(iter_record_paths(records_root))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {records_root}")
    if limit is not None:
        mat_files = mat_files[:limit]

    records, signals = [], []
    for mat_path in mat_files:
        signal = extract_signal_matrix(loadmat(mat_path))
        if signal.ndim != 2:
            raise ValueError(f"Expected a 2D signal matrix, got {signal.shape} for {mat_path}")
        dx_codes = extract_dx_codes(mat_path.with_suffix(".hea"))
        records.append(
            {
                "record_id": mat_path.stem,
                "relative_path": str(mat_path.relative_to(records_root)),
                "n_leads": int(signal.shape[0]),
                "n_samples": int(signal.shape[1]),
                "dx_codes": ",".join(dx_codes),
                "n_labels": len(dx_codes),
            }
        )
        signals.append(signal)

    return EcgDataset(signals=_pad_to_tensor(signals), metadata=pd.DataFrame(records))
