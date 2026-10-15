"""Per-cluster multi-classifier training for stress and burnout targets."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


MODELS_DIR = "outputs/models"


def _get_classifiers(n_classes: int) -> Dict[str, Any]:
    """Instantiate the three classifier types with a fixed random state.

    Parameters
    ----------
    n_classes : int
        Number of target classes in this training fold — selects the XGBoost
        eval metric (``logloss`` binary, ``mlogloss`` multi-class).
    """
    xgb_metric = "logloss" if n_classes == 2 else "mlogloss"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lr = LogisticRegression(max_iter=1000, random_state=42)
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric=xgb_metric,
            verbosity=0,
        ),
        "LogisticRegression": lr,
    }


def train_per_cluster_classifiers(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    save_dir: str = MODELS_DIR,
) -> Tuple[Dict[int, Dict[str, Dict]], Dict[int, Dict[str, Any]]]:
    """Train three classifiers on each cluster's data for a given target.

    XGBoost requires contiguous integer labels starting from 0.  When a
    cluster is missing one of the global classes, labels are re-encoded
    locally for XGBoost and predictions are mapped back to global codes.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed DataFrame with a ``cluster`` column.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Encoded target column (``stress_level_encoded`` or ``burnout_encoded``).
    save_dir : str
        Directory for saving trained model ``.pkl`` files.

    Returns
    -------
    all_results : dict
        ``{cluster_id: {model_name: metrics_dict}}``.
    all_models : dict
        ``{cluster_id: {model_name: fitted_model}}``.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    clusters = sorted(df["cluster"].unique())
    all_results: Dict[int, Dict[str, Dict]] = {}
    all_models: Dict[int, Dict[str, Any]] = {}

    for cid in clusters:
        print(f"\n[classification] -- Cluster {cid} | target={target_col} --")
        sub = df[df["cluster"] == cid].copy()

        if len(sub) < 20:
            print(f"[classification] Skipped cluster {cid}: too few rows ({len(sub)})")
            continue

        X = sub[feature_cols]
        y_global = sub[target_col]
        n_classes = y_global.nunique()

        # Re-encode labels to contiguous 0..n-1 (needed for XGBoost when
        # this cluster is missing some global classes)
        local_le = LabelEncoder()
        y_local = pd.Series(local_le.fit_transform(y_global), index=y_global.index)
        global_classes = local_le.classes_  # maps local -> global label

        # One-class cluster: use a trivial predictor so the hybrid can route
        # samples correctly instead of falling back to the global mode.
        if n_classes < 2:
            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            X_tr, X_te, y_tr, y_te_local = train_test_split(
                X, y_local, test_size=0.2, random_state=42
            )
            y_te_global = y_global.loc[y_te_local.index]
            dummy.fit(X_tr, y_tr)
            y_pred_local = dummy.predict(X_te)
            y_pred_global = global_classes[y_pred_local]
            acc = accuracy_score(y_te_global, y_pred_global)
            macro_f1 = f1_score(
                y_te_global, y_pred_global, average="macro", zero_division=0
            )
            single_cls = int(global_classes[0])
            print(
                f"[classification] Cluster {cid}: pure cluster "
                f"(class={single_cls}) - trivial predictor  "
                f"acc={acc:.3f}  macro_f1={macro_f1:.3f}"
            )
            model_path = (
                Path(save_dir) / f"cluster{cid}_DummyClassifier_{target_col}.pkl"
            )
            joblib.dump({"model": dummy, "local_le": local_le}, model_path)
            all_results[cid] = {
                "DummyClassifier": {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "y_test": y_te_global,
                    "y_pred": y_pred_global,
                    "local_le": local_le,
                }
            }
            all_models[cid] = {"DummyClassifier": dummy}
            continue

        # Stratified split with graceful fallback
        try:
            min_class_count = y_local.value_counts().min()
            use_stratify = min_class_count >= 2
            X_tr, X_te, y_tr, y_te_local = train_test_split(
                X, y_local, test_size=0.2, random_state=42,
                stratify=y_local if use_stratify else None,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te_local = train_test_split(
                X, y_local, test_size=0.2, random_state=42
            )

        # True test labels in global encoding (for metrics)
        y_te_global = y_global.loc[y_te_local.index]

        # After splitting, training portion may still have only one class when
        # the cluster is highly skewed (minority class went entirely to test).
        # Real classifiers would crash or silently trivialise — use DummyClassifier.
        if y_tr.nunique() < 2:
            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy.fit(X_tr, y_tr)
            y_pred_local = dummy.predict(X_te)
            y_pred_global = global_classes[y_pred_local]
            acc = accuracy_score(y_te_global, y_pred_global)
            macro_f1 = f1_score(
                y_te_global, y_pred_global, average="macro", zero_division=0
            )
            single_cls = int(global_classes[int(y_tr.iloc[0])])
            print(
                f"[classification] Cluster {cid}: train split has one class "
                f"(class={single_cls}) - trivial predictor  "
                f"acc={acc:.3f}  macro_f1={macro_f1:.3f}"
            )
            model_path = (
                Path(save_dir) / f"cluster{cid}_DummyClassifier_{target_col}.pkl"
            )
            joblib.dump({"model": dummy, "local_le": local_le}, model_path)
            all_results[cid] = {
                "DummyClassifier": {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "y_test": y_te_global,
                    "y_pred": y_pred_global,
                    "local_le": local_le,
                }
            }
            all_models[cid] = {"DummyClassifier": dummy}
            continue

        classifiers = _get_classifiers(n_classes)
        cluster_results: Dict[str, Dict] = {}
        cluster_models: Dict[str, Any] = {}

        for name, clf in classifiers.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_tr, y_tr)

                y_pred_local = clf.predict(X_te)
                # Map local predictions back to global label space
                y_pred_global = global_classes[y_pred_local]

                acc = accuracy_score(y_te_global, y_pred_global)
                macro_f1 = f1_score(
                    y_te_global, y_pred_global, average="macro", zero_division=0
                )
                report = classification_report(
                    y_te_global, y_pred_global, output_dict=True, zero_division=0
                )

                cluster_results[name] = {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "report": report,
                    "y_test": y_te_global,
                    "y_pred": y_pred_global,
                    "local_le": local_le,   # stored so HybridModel can invert
                }
                cluster_models[name] = clf

                model_path = Path(save_dir) / f"cluster{cid}_{name}_{target_col}.pkl"
                joblib.dump({"model": clf, "local_le": local_le}, model_path)
                print(
                    f"[classification] {name:<20} acc={acc:.3f}  macro_f1={macro_f1:.3f}"
                )
            except Exception as exc:
                print(f"[classification] {name} failed on cluster {cid}: {exc}")

        all_results[cid] = cluster_results
        all_models[cid] = cluster_models

    return all_results, all_models
