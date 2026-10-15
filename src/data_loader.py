"""Dataset loading with schema normalisation for multiple student-stress CSV formats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying",
    "stress_level", "burnout",
]

# Numeric ranges for each feature (used in synthetic generation)
_FEAT_RANGES = {
    "anxiety_level":              (0,  21),
    "self_esteem":                (0,  30),
    "mental_health_history":      (0,   1),
    "depression":                 (0,  27),
    "headache":                   (0,   5),
    "blood_pressure":             (1,   3),
    "sleep_quality":              (0,   5),
    "breathing_problem":          (0,   5),
    "noise_level":                (0,   5),
    "living_conditions":          (0,   5),
    "safety":                     (0,   5),
    "basic_needs":                (0,   5),
    "academic_performance":       (0,   5),
    "study_load":                 (0,   5),
    "teacher_student_relationship":(0,  5),
    "future_career_concerns":     (0,   5),
    "social_support":             (0,   5),
    "peer_pressure":              (0,   5),
    "extracurricular_activities": (0,   5),
    "bullying":                   (0,   5),
}

_STRESS_MAP = {0: "Low", 1: "Medium", 2: "High"}


def _derive_burnout(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Derive a burnout label from depression, anxiety, stress, and study load.

    Uses a threshold on a weighted combination so burnout is correlated with
    but not identical to stress_level, giving it value as a separate target.
    """
    stress_int = df["stress_level"].map({"Low": 0, "Medium": 1, "High": 2}).values
    score = (
        (df["depression"].values / 27.0) * 0.40
        + (df["anxiety_level"].values / 21.0) * 0.30
        + (stress_int / 2.0) * 0.20
        + (df["study_load"].values / 5.0) * 0.10
        + rng.normal(0, 0.05, len(df))
    )
    return np.where(np.clip(score, 0, 1) > 0.55, "Yes", "No")


def _normalize_stress_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the StressLevelDataset (rxnach) schema.

    - Converts integer stress_level (0/1/2) to string labels (Low/Medium/High).
    - Derives a burnout column from psychometric features.
    """
    df = df.copy()
    df["stress_level"] = df["stress_level"].astype(int).map(_STRESS_MAP)
    rng = np.random.default_rng(42)
    df["burnout"] = _derive_burnout(df, rng)
    return df


def _normalize_lifestyle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the student-lifestyle CSV (Sleep_Hours_Per_Day schema)."""
    _COL_MAP = {
        "Sleep_Hours_Per_Day": "sleep_hours",
        "Study_Hours_Per_Day": "study_hours",
        "Extracurricular_Hours_Per_Day": "extracurricular_hours",
        "Social_Hours_Per_Day": "social_hours",
        "Physical_Activity_Hours_Per_Day": "physical_activity",
        "GPA": "academic_performance",
        "Stress_Level": "stress_level",
    }
    df = df.drop(columns=[c for c in ["Student_ID"] if c in df.columns])
    df = df.rename(columns=_COL_MAP)
    df["stress_level"] = df["stress_level"].replace({"Moderate": "Medium"})
    rng = np.random.default_rng(42)
    max_pa = max(df["physical_activity"].max(), 1.0)
    score = np.clip(
        (df["study_hours"].values / 12.0) * 0.40
        + (1 - df["sleep_hours"].values / 10.0) * 0.40
        + (1 - df["physical_activity"].values / max_pa) * 0.20
        + rng.normal(0, 0.06, len(df)),
        0, 1,
    )
    df["burnout"] = np.where(score > 0.52, "Yes", "No")
    return df


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset matching the StressLevelDataset schema.

    Used as a fallback when no CSV is available.
    """
    rng = np.random.default_rng(random_state)
    rows = {}
    for col, (lo, hi) in _FEAT_RANGES.items():
        if col == "mental_health_history":
            rows[col] = rng.integers(0, 2, n_samples)
        elif col == "blood_pressure":
            rows[col] = rng.integers(1, 4, n_samples)
        else:
            rows[col] = rng.integers(lo, hi + 1, n_samples)

    stress_raw = np.clip(
        rows["anxiety_level"] / 21.0 * 0.25
        + rows["depression"] / 27.0 * 0.25
        + rows["study_load"] / 5.0 * 0.20
        + (1 - rows["self_esteem"] / 30.0) * 0.15
        + (1 - rows["social_support"] / 5.0) * 0.15
        + rng.normal(0, 0.07, n_samples),
        0, 1,
    )
    stress_int = np.where(stress_raw < 0.38, 0, np.where(stress_raw < 0.68, 1, 2))
    rows["stress_level"] = [_STRESS_MAP[v] for v in stress_int]

    df = pd.DataFrame(rows)
    rng2 = np.random.default_rng(random_state + 1)
    df["burnout"] = _derive_burnout(df, rng2)
    return df


def load_data(filepath: Optional[str] = None, n_samples: int = 2000) -> pd.DataFrame:
    """Load and normalise a student stress dataset.

    Handles three schemas:
    - StressLevelDataset (anxiety_level, depression, ... stress_level 0/1/2)
    - student_lifestyle_dataset (Sleep_Hours_Per_Day, GPA, ... Stress_Level)
    - Synthetic fallback

    Parameters
    ----------
    filepath : str or None
    n_samples : int  (synthetic fallback only)

    Returns
    -------
    pd.DataFrame conforming to EXPECTED_COLUMNS
    """
    if filepath is not None and Path(filepath).exists():
        try:
            df = pd.read_csv(filepath)
            print(f"[data_loader] Loaded {len(df):,} rows from: {filepath}")

            if "anxiety_level" in df.columns:
                print("[data_loader] Detected StressLevelDataset schema - normalising...")
                df = _normalize_stress_dataset(df)
            elif "Sleep_Hours_Per_Day" in df.columns:
                print("[data_loader] Detected student-lifestyle schema - normalising...")
                df = _normalize_lifestyle_dataset(df)
            else:
                missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
                if missing:
                    raise ValueError(f"Unrecognised schema. Missing: {missing}")

        except Exception as exc:
            print(f"[data_loader] CSV load failed - {exc}")
            print("[data_loader] Falling back to synthetic dataset...")
            df = generate_synthetic_data(n_samples)
    else:
        print("[data_loader] No valid CSV path - generating synthetic dataset...")
        df = generate_synthetic_data(n_samples)

    print(f"[data_loader] Shape    : {df.shape}")
    print(f"[data_loader] Columns  : {list(df.columns)}")
    print(f"[data_loader] stress_level : {df['stress_level'].value_counts().to_dict()}")
    print(f"[data_loader] burnout      : {df['burnout'].value_counts().to_dict()}")
    print(f"\n{df.describe().to_string()}\n")
    return df
