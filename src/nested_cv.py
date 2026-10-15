"""Outer nested cross-validation for the full hybrid pipeline.

Each fold re-fits the entire pipeline from scratch:
  imputer + capper -> scaler -> KMeans -> per-cluster classifiers -> HybridModel

This gives an unbiased macro-F1 estimate that accounts for every model-selection
decision made during training (cluster assignment, specialist choice, etc.).
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from classification import train_per_cluster_classifiers
from hybrid_model import HybridModel
from preprocessing import (
    apply_encoding_and_scaling,
    fit_transform_cleaning,
    transform_with_scaler,
)


def nested_cv_hybrid(
    df: pd.DataFrame,
    feature_cols: List[str],
    base_artifacts: Dict,
    target_col: str = "stress_level_encoded",
    optimal_k: int = 4,
    n_splits: int = 5,
    n_repeats: int = 2,
) -> np.ndarray:
    """Run outer repeated stratified K-fold CV for the full hybrid pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Training-set-only DataFrame (post feature engineering, pre-scaling).
    feature_cols : list[str]
    base_artifacts : dict
        Stateless artifacts from preprocess_data: le_stress, le_burnout,
        numeric_cols.  Scaler / capper / imputer are NOT passed in — they are
        re-fitted from scratch in every fold to prevent leakage.
    target_col : str
    optimal_k : int
        K for K-Means, pre-selected by silhouette on the full training set.
        Fixing K across folds keeps inter-fold variation attributable to data
        partition rather than K selection.
    n_splits, n_repeats : int
        Outer CV grid — default 5x2 = 10 folds.

    Returns
    -------
    np.ndarray  Per-fold macro-F1 scores (length = n_splits * n_repeats).
    """
    rscv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    y = df[target_col]
    scores: List[float] = []
    total = n_splits * n_repeats

    for fold_idx, (train_idx, val_idx) in enumerate(rscv.split(df, y), start=1):
        print(f"[nested_cv] Fold {fold_idx}/{total} ...", end=" ", flush=True)

        fold_train = df.iloc[train_idx].copy().reset_index(drop=True)
        fold_val   = df.iloc[val_idx].copy().reset_index(drop=True)

        # Only pass stateless artifacts — everything else is re-fitted below
        fold_artifacts: Dict = {
            "le_stress":    base_artifacts["le_stress"],
            "le_burnout":   base_artifacts["le_burnout"],
            "numeric_cols": base_artifacts["numeric_cols"],
        }

        # Silence inner pipeline prints; only fold-level progress is shown
        with contextlib.redirect_stdout(io.StringIO()):
            fold_train, fold_val = fit_transform_cleaning(fold_train, fold_val, fold_artifacts)
            fold_train, _        = apply_encoding_and_scaling(fold_train, fold_artifacts)
            fold_val             = transform_with_scaler(fold_val, fold_artifacts)

            fcols = [c for c in feature_cols if c in fold_train.columns]

            km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            fold_train = fold_train.copy()
            fold_train["cluster"] = km.fit_predict(fold_train[fcols])
            fold_val = fold_val.copy()
            fold_val["cluster"]   = km.predict(fold_val[fcols])

            fold_results, fold_models = train_per_cluster_classifiers(
                fold_train, fcols, target_col=target_col
            )

            fold_hybrid = HybridModel(km, fold_models, fcols)
            fold_hybrid.select_best_per_cluster(fold_results)
            y_pred = fold_hybrid.predict(fold_val)

        macro_f1 = float(
            f1_score(fold_val[target_col], y_pred, average="macro", zero_division=0)
        )
        scores.append(macro_f1)
        print(f"macro_f1={macro_f1:.4f}")

    return np.array(scores)
