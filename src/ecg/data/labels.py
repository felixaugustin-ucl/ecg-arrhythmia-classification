"""SNOMED-CT diagnosis codes to human-readable condition labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: Codes whose published label is wrong or unhelpfully abbreviated.
MANUAL_OVERRIDES = {
    "67741000119109": "Left Atrial Enlargement (Disorder)",
    "67751000119106": "Right Atrial High Voltage",
}

_TRAILING_FLOAT = re.compile(r"\d+\.0")
_ECG_PREFIX = re.compile(r"^Electrocardiogram:\s*", re.IGNORECASE)
_CATEGORY_SUFFIX = re.compile(r"\s*\((Disorder|Finding)\)\s*$", re.IGNORECASE)


def normalize_code(code: object) -> str:
    """Strip a trailing ``.0`` left behind when codes round-trip through floats."""
    text = str(code).strip()
    if not text or text.lower() == "nan":
        return ""
    return text[:-2] if _TRAILING_FLOAT.fullmatch(text) else text


def to_title_case(label: object) -> str:
    """Title-case a condition name and drop presentation wrappers."""
    cleaned = " ".join(str(label).split()).strip()
    if not cleaned:
        return ""
    titled = cleaned.title()
    titled = _ECG_PREFIX.sub("", titled).strip()
    return _CATEGORY_SUFFIX.sub("", titled).strip()


def parse_code_list(raw: object) -> list[str]:
    """Split a comma-separated code string, dropping blanks."""
    return [code for code in (normalize_code(c) for c in str(raw).split(",")) if code]


def _prepare_lookup(df: pd.DataFrame, code_col: str, label_col: str) -> pd.DataFrame:
    """Coerce a lookup file into the shared ``dx_code``/``condition_label`` schema."""
    out = df.rename(columns={code_col: "dx_code", label_col: "condition_label"}).copy()
    out["dx_code"] = out["dx_code"].apply(normalize_code)
    out["condition_label"] = out["condition_label"].astype(str).str.strip().apply(to_title_case)
    return out[["dx_code", "condition_label"]]


def build_condition_lookup(primary_csv: Path, remaining_csv: Path) -> dict[str, str]:
    """Merge both SNOMED label files into one code to label mapping.

    The primary file wins on conflict; :data:`MANUAL_OVERRIDES` wins overall.
    """
    frames = [
        _prepare_lookup(pd.read_csv(primary_csv), "Snomed_CT", "Full Name"),
        _prepare_lookup(pd.read_csv(remaining_csv), "dx_code", "preferred_label"),
    ]
    lookup = pd.concat(frames, ignore_index=True)
    lookup = lookup[
        lookup["dx_code"].ne("")
        & lookup["condition_label"].ne("")
        & lookup["condition_label"].str.lower().ne("nan")
    ].drop_duplicates(subset=["dx_code"], keep="first")

    mapping = dict(zip(lookup["dx_code"], lookup["condition_label"], strict=True))
    mapping.update({code: to_title_case(label) for code, label in MANUAL_OVERRIDES.items()})
    return mapping


def attach_conditions(metadata: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    """Add ``dx_code_list``/``dx_condition_list``/``dx_conditions`` columns."""
    out = metadata.copy()
    out["dx_code_list"] = out["dx_codes"].fillna("").astype(str).apply(parse_code_list)
    out["dx_condition_list"] = out["dx_code_list"].apply(
        lambda codes: [
            to_title_case(lookup.get(code, f"Unknown Condition ({code})")) for code in codes
        ]
    )
    out["dx_conditions"] = out["dx_condition_list"].str.join(", ")
    return out


def label_sets(metadata: pd.DataFrame) -> pd.Series:
    """Deduplicated, sorted code sets — the input to ``MultiLabelBinarizer``."""
    return metadata["dx_code_list"].apply(
        lambda codes: sorted({str(code).strip() for code in codes if str(code).strip()})
    )
