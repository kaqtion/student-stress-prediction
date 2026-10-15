"""Preprocessing pipeline: dedup, imputation, outlier capping, encoding, scaling.

Stateless operations (dedup, target encoding) run on the full dataset before
the train/test split.  Statistics-fitting operations (imputer, outlier capper,
MinMaxScaler) are fitted on the training set only to prevent data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# All 20 psychometric feature columns from StressLevelDataset
NUMERIC_COLS: List[str] = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying",
]

TARGET_STRESS = "stress_level"
TARGET_BURNOUT = "burnout"


class OutlierCapper:
    """Cap numeric outliers at IQR fences without dropping any rows."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "OutlierCapper":
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in self.bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Stateless preprocessing: dedup and target label encoding only.

    These operations require no statistics learned from data and are safe
    to run on the full dataset before the train/test split.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Label-encode the two targets (stress_level -> 0/1/2, burnout -> 0/1).

    Note: mean imputation and outlier capping are intentionally deferred to
    ``fit_transform_cleaning`` so they are fitted on training data only.
    """
    print("[preprocessing] Starting preprocessing pipeline...")
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[preprocessing] Duplicate rows removed: {before - len(df)}")

    le_stress = LabelEncoder()
    le_burnout = LabelEncoder()
    df["stress_level_encoded"] = le_stress.fit_transform(df[TARGET_STRESS])
    df["burnout_encoded"] = le_burnout.fit_transform(df[TARGET_BURNOUT])
    print(
        f"[preprocessing] stress_level classes : {list(le_stress.classes_)} -> "
        f"{list(range(len(le_stress.classes_)))}"
    )
    print(
        f"[preprocessing] burnout classes      : {list(le_burnout.classes_)} -> "
        f"{list(range(len(le_burnout.classes_)))}"
    )

    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    artifacts: Dict = {
        "le_stress": le_stress,
        "le_burnout": le_burnout,
        "numeric_cols": present_numeric,
    }
    print(f"[preprocessing] Done - shape: {df.shape}")
    return df, artifacts


def fit_transform_cleaning(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    artifacts: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit imputer and outlier capper on train set; apply to both train and test.

    Must be called AFTER the global train/test split so no test-set statistics
    influence the fitted parameters (imputer means, IQR fences).

    Parameters
    ----------
    df_train : pd.DataFrame  (880 rows, training set)
    df_test  : pd.DataFrame  (220 rows, held-out test set)
    artifacts : dict         (updated in place with capper and impute_means)

    Returns
    -------
    df_train, df_test : DataFrames with NaNs filled and outliers capped.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    present_numeric = artifacts["numeric_cols"]

    # Impute: compute column means from training rows only
    impute_means: Dict[str, float] = {}
    for col in present_numeric:
        if df_train[col].isnull().any():
            impute_means[col] = float(df_train[col].mean())
            df_train[col] = df_train[col].fillna(impute_means[col])
    for col, mean_val in impute_means.items():
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(mean_val)

    # Outlier capping: IQR fences from training rows only
    capper = OutlierCapper(factor=1.5)
    capper.fit(df_train, present_numeric)
    df_train = capper.transform(df_train)
    df_test = capper.transform(df_test)

    artifacts.update({"capper": capper, "impute_means": impute_means})
    print(
        f"[preprocessing] Imputer + OutlierCapper fitted on train ({len(df_train)} rows), "
        f"applied to test ({len(df_test)} rows)."
    )
    return df_train, df_test


def apply_encoding_and_scaling(
    df: pd.DataFrame, artifacts: Dict
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Fit MinMaxScaler on *df* (training set) and transform it.

    Parameters
    ----------
    df : pd.DataFrame  (training set only)
    artifacts : dict   (updated in place with scaler and scale_cols)
    """
    print("[preprocessing] Applying MinMaxScaling...")
    df = df.copy()

    skip_set = {
        TARGET_STRESS, TARGET_BURNOUT,
        "stress_level_encoded", "burnout_encoded", "cluster",
    }
    scale_cols = [
        c for c in df.columns
        if c not in skip_set and pd.api.types.is_numeric_dtype(df[c])
    ]

    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    artifacts.update({"scaler": scaler, "scale_cols": scale_cols})
    print(f"[preprocessing] Scaled {len(scale_cols)} columns.")
    return df, scaler


def transform_with_scaler(df: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
    """Apply the already-fitted MinMaxScaler to *df* (test-set transform)."""
    df = df.copy()
    scaler = artifacts["scaler"]
    scale_cols = [c for c in artifacts["scale_cols"] if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])
    print(f"[preprocessing] Test set transformed with fitted scaler ({len(scale_cols)} columns).")
    return df
