"""Feature engineering for the StressLevelDataset psychometric schema."""

from __future__ import annotations

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add five domain-derived composite features to *df*.

    All formulas use raw (pre-MinMaxScaler) column values.

    New columns
    -----------
    psychological_burden
        Weighted combination of core mental-health indicators.  High anxiety
        and depression with low self-esteem signals overall psychological load.
    academic_stress
        Study pressure minus academic performance; captures the gap between
        effort demanded and output achieved.
    wellbeing_index
        Composite of protective factors (sleep, social support, safety) minus
        physical stress symptoms (headache, breathing problems).
    environmental_stress
        External stressors: noise, bullying, minus quality of living conditions.
    support_deficit
        Demand-supply gap in social and academic support systems.

    Parameters
    ----------
    df : pd.DataFrame  (must contain the 20 base psychometric columns)

    Returns
    -------
    pd.DataFrame with five additional columns.
    """
    print("[feature_engineering] Creating engineered features...")
    df = df.copy()

    # Psychological load from core mental-health scales
    df["psychological_burden"] = (
        df["anxiety_level"] / 21.0 * 0.35
        + df["depression"] / 27.0 * 0.35
        - df["self_esteem"] / 30.0 * 0.30
    )

    # Academic pressure relative to achieved performance
    df["academic_stress"] = (
        df["study_load"] * 0.40
        + df["peer_pressure"] * 0.30
        - df["academic_performance"] * 0.30
    )

    # Protective wellness factors minus physical stress symptoms
    df["wellbeing_index"] = (
        df["sleep_quality"]
        + df["social_support"]
        + df["safety"]
        + df["basic_needs"]
        - df["headache"]
        - df["breathing_problem"]
    )

    # External / environmental stressors
    df["environmental_stress"] = (
        df["noise_level"] * 0.40
        + df["bullying"] * 0.40
        - df["living_conditions"] * 0.20
    )

    # Gap between demands and available support
    df["support_deficit"] = (
        df["peer_pressure"]
        + df["future_career_concerns"]
        - df["social_support"]
        - df["teacher_student_relationship"]
    )

    engineered = [
        "psychological_burden", "academic_stress", "wellbeing_index",
        "environmental_stress", "support_deficit",
    ]
    print(f"[feature_engineering] Added: {engineered}")
    print(f"[feature_engineering] DataFrame shape: {df.shape}")
    return df
