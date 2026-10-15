"""Hybrid model: KMeans cluster assignment -> per-cluster best classifier."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


MODELS_DIR = "outputs/models"


class HybridModel:
    """Cluster-then-classify pipeline.

    For each cluster the highest macro-F1 specialist is selected and used
    at prediction time, regardless of its internal evaluation score.  This
    preserves the genuinely distinct routing behaviour that differentiates
    the hybrid from the global baseline RF.

    A fallback model is kept only as a safety net for clusters that had no
    trained specialist at all (e.g. clusters skipped due to too few samples).
    """

    def __init__(
        self,
        kmeans: KMeans,
        cluster_models: Dict[int, Dict[str, Any]],
        feature_cols: List[str],
        fallback_model: Optional[Any] = None,
    ) -> None:
        self.kmeans = kmeans
        self.cluster_models = cluster_models
        self.feature_cols = feature_cols
        self.fallback_model = fallback_model
        self.best_models: Dict[int, Tuple[str, Any]] = {}
        self.fallback_clusters: set = set()

    def select_best_per_cluster(self, results: Dict[int, Dict[str, Dict]]) -> None:
        """Choose the highest macro-F1 specialist for every cluster.

        Only clusters with no results at all fall back to the global model.
        """
        for cid, cluster_res in results.items():
            if not cluster_res:
                self.fallback_clusters.add(cid)
                print(f"[hybrid] Cluster {cid}: no trained model -> safety-net fallback")
                continue
            best_name = max(cluster_res, key=lambda n: cluster_res[n]["macro_f1"])
            best_score = cluster_res[best_name]["macro_f1"]
            raw_model = self.cluster_models[cid][best_name]
            local_le = cluster_res[best_name].get("local_le", None)
            self.best_models[cid] = (best_name, (raw_model, local_le))
            print(
                f"[hybrid] Cluster {cid}: specialist = {best_name}  "
                f"(internal macro_f1={best_score:.3f})"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Route each sample to its cluster specialist."""
        clusters = self.kmeans.predict(X[self.feature_cols])
        predictions = np.full(len(X), -1, dtype=float)

        for cid, (model_name, model_info) in self.best_models.items():
            mask = clusters == cid
            if mask.sum() == 0:
                continue
            X_sub = X[self.feature_cols].iloc[np.where(mask)[0]]

            if isinstance(model_info, tuple):
                model, local_le = model_info
            else:
                model, local_le = model_info, None

            local_preds = model.predict(X_sub)
            global_preds = local_le.classes_[local_preds] if local_le is not None else local_preds
            predictions[mask] = global_preds

        # Safety net: clusters with no specialist -> global fallback
        if self.fallback_model is not None and self.fallback_clusters:
            for cid in self.fallback_clusters:
                mask = clusters == cid
                if mask.sum() == 0:
                    continue
                X_sub = X[self.feature_cols].iloc[np.where(mask)[0]]
                predictions[mask] = self.fallback_model.predict(X_sub)

        # Final safety net: mode-fill anything still unrouted
        unfilled = predictions < 0
        if unfilled.any():
            filled = predictions[~unfilled]
            mode_val = int(pd.Series(filled).mode().iloc[0]) if len(filled) > 0 else 0
            predictions[unfilled] = mode_val

        return predictions.astype(int)

    def save(self, path: str = f"{MODELS_DIR}/hybrid_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[hybrid] Model saved -> {path}")


def _repeated_cv(estimator, X, y, n_splits: int = 5, n_repeats: int = 3) -> np.ndarray:
    """5x3 repeated stratified K-fold CV returning per-fold macro-F1 scores."""
    rscv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    return cross_val_score(estimator, X, y, cv=rscv, scoring="f1_macro", n_jobs=-1)


def train_global_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    save_dir: str = MODELS_DIR,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train a global Random Forest baseline on *df_train*, evaluate on *df_test*.

    Uses 5x3 repeated stratified K-fold CV on the training set for a robust
    performance estimate (15 folds instead of 5).
    """
    print("[hybrid] Training global baseline Random Forest...")
    X_tr, y_tr = df_train[feature_cols], df_train[target_col]
    X_te, y_te = df_test[feature_cols], df_test[target_col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    mac_p = precision_score(y_te, y_pred, average="macro", zero_division=0)
    mac_r = recall_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"[hybrid] Baseline RF  acc={acc:.4f}  bal_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}")
    print(classification_report(y_te, y_pred, zero_division=0))

    cv_f1 = _repeated_cv(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_tr, y_tr
    )
    print(
        f"[hybrid] Baseline RF  5x3 repeated CV macro-F1: "
        f"{cv_f1.mean():.4f} +/- {cv_f1.std():.4f}"
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, f"{save_dir}/baseline_rf_{target_col}.pkl")

    return rf, {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": mac_p,
        "macro_recall": mac_r,
        "macro_f1": macro_f1,
        "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_macro_f1_std": round(float(cv_f1.std()), 4),
        "y_test": y_te,
        "y_pred": y_pred,
    }


def train_extra_baselines(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    save_dir: str = MODELS_DIR,
) -> Dict[str, Tuple[Any, Dict]]:
    """Train a tuned XGBoost and a Logistic Regression baseline.

    Both are evaluated on *df_test* and assessed with 5x3 repeated
    stratified K-fold CV on *df_train*.

    Returns
    -------
    dict  {model_name: (fitted_model, metrics_dict)}
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = df_train[feature_cols], df_train[target_col]
    X_te, y_te = df_test[feature_cols], df_test[target_col]
    n_cls = int(y_tr.nunique())
    results: Dict[str, Tuple[Any, Dict]] = {}

    # ── XGBoost with RandomizedSearchCV ──────────────────────────────────────
    print("[hybrid] Tuning XGBoost baseline (RandomizedSearchCV n_iter=15, cv=3)...")
    xgb_metric = "logloss" if n_cls == 2 else "mlogloss"
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs = RandomizedSearchCV(
            XGBClassifier(random_state=42, eval_metric=xgb_metric, verbosity=0),
            param_dist,
            n_iter=15,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        rs.fit(X_tr, y_tr)

    xgb_best = rs.best_estimator_
    print(f"[hybrid] XGBoost best params: {rs.best_params_}")
    y_pred_xgb = xgb_best.predict(X_te)
    acc_xgb = accuracy_score(y_te, y_pred_xgb)
    bal_xgb = balanced_accuracy_score(y_te, y_pred_xgb)
    f1_xgb = f1_score(y_te, y_pred_xgb, average="macro", zero_division=0)
    cv_xgb = _repeated_cv(xgb_best, X_tr, y_tr)
    print(f"[hybrid] XGBoost (tuned)  acc={acc_xgb:.4f}  bal_acc={bal_xgb:.4f}  macro_f1={f1_xgb:.4f}")
    print(f"[hybrid] XGBoost (tuned)  5x3 CV macro-F1: {cv_xgb.mean():.4f} +/- {cv_xgb.std():.4f}")
    print(classification_report(y_te, y_pred_xgb, zero_division=0))
    joblib.dump(xgb_best, f"{save_dir}/baseline_xgb_{target_col}.pkl")
    results["XGBoost_tuned"] = (xgb_best, {
        "accuracy": acc_xgb,
        "balanced_accuracy": bal_xgb,
        "macro_f1": f1_xgb,
        "cv_macro_f1_mean": round(float(cv_xgb.mean()), 4),
        "cv_macro_f1_std": round(float(cv_xgb.std()), 4),
        "y_test": y_te,
        "y_pred": y_pred_xgb,
    })

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("[hybrid] Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
    acc_lr = accuracy_score(y_te, y_pred_lr)
    bal_lr = balanced_accuracy_score(y_te, y_pred_lr)
    f1_lr = f1_score(y_te, y_pred_lr, average="macro", zero_division=0)
    cv_lr = _repeated_cv(
        LogisticRegression(max_iter=2000, random_state=42), X_tr, y_tr
    )
    print(f"[hybrid] Logistic Reg      acc={acc_lr:.4f}  bal_acc={bal_lr:.4f}  macro_f1={f1_lr:.4f}")
    print(f"[hybrid] Logistic Reg      5x3 CV macro-F1: {cv_lr.mean():.4f} +/- {cv_lr.std():.4f}")
    print(classification_report(y_te, y_pred_lr, zero_division=0))
    joblib.dump(lr, f"{save_dir}/baseline_lr_{target_col}.pkl")
    results["LogisticReg"] = (lr, {
        "accuracy": acc_lr,
        "balanced_accuracy": bal_lr,
        "macro_f1": f1_lr,
        "cv_macro_f1_mean": round(float(cv_lr.mean()), 4),
        "cv_macro_f1_std": round(float(cv_lr.std()), 4),
        "y_test": y_te,
        "y_pred": y_pred_lr,
    })

    return results


def evaluate_hybrid(
    hybrid: HybridModel,
    df_test: pd.DataFrame,
    target_col: str,
) -> Dict:
    """Evaluate the hybrid model on the provided held-out test set."""
    print("[hybrid] Evaluating hybrid model on held-out test set...")
    y_te = df_test[target_col]
    y_pred = hybrid.predict(df_test)

    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"[hybrid] Hybrid model  acc={acc:.4f}  bal_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}")
    print(classification_report(y_te, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "cv_macro_f1_mean": 0.0,
        "cv_macro_f1_std": 0.0,
        "y_test": y_te,
        "y_pred": y_pred,
    }


def ablation_cluster_breakdown(
    hybrid: HybridModel,
    pred_dict: Dict[str, Any],
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Per-cluster accuracy breakdown: hybrid vs every baseline.

    Parameters
    ----------
    pred_dict : {model_name: predictions_array}
        Pre-computed predictions for every model on df_test.

    Returns
    -------
    pd.DataFrame  one row per cluster, columns: cluster, n_test,
                  dominant_class, class_purity, acc_<model> for each model.
    """
    clusters = hybrid.kmeans.predict(df_test[feature_cols])
    y_true = np.array(df_test[target_col])

    rows = []
    for cid in sorted(np.unique(clusters)):
        mask = clusters == cid
        n = int(mask.sum())
        y_sub = y_true[mask]

        row: Dict[str, Any] = {
            "cluster": int(cid),
            "n_test": n,
            "dominant_class": int(pd.Series(y_sub).mode().iloc[0]),
            "class_purity": round(float(pd.Series(y_sub).value_counts().iloc[0] / n), 3),
        }
        for model_name, preds in pred_dict.items():
            row[f"acc_{model_name}"] = round(
                float(accuracy_score(y_sub, np.array(preds)[mask])), 4
            )
        rows.append(row)

    return pd.DataFrame(rows)


def train_class_baselines(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    save_dir: str = MODELS_DIR,
) -> Dict[str, Tuple[Any, Dict]]:
    """Train the four class-study baselines: DT, KNN, Naive Bayes, SVM.

    Each model is evaluated on *df_test* and assessed with 5x3 repeated
    stratified K-fold CV on *df_train*.

    Returns
    -------
    dict  {model_name: (fitted_model, metrics_dict)}
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = df_train[feature_cols], df_train[target_col]
    X_te, y_te = df_test[feature_cols], df_test[target_col]
    results: Dict[str, Tuple[Any, Dict]] = {}

    candidates = [
        ("DecisionTree",  DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("KNN",           KNeighborsClassifier(n_neighbors=5)),
        ("NaiveBayes",    GaussianNB()),
        ("SVM",           SVC(kernel="rbf", C=1.0, random_state=42)),
    ]

    for name, model in candidates:
        print(f"[hybrid] Training {name}...")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc     = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        mac_f1  = f1_score(y_te, y_pred, average="macro",     zero_division=0)
        mac_p   = precision_score(y_te, y_pred, average="macro", zero_division=0)
        mac_r   = recall_score(y_te, y_pred, average="macro",  zero_division=0)

        cv_f1 = _repeated_cv(model, X_tr, y_tr)
        print(
            f"[hybrid] {name:<14}  acc={acc:.4f}  bal_acc={bal_acc:.4f}"
            f"  macro_f1={mac_f1:.4f}"
        )
        print(f"[hybrid] {name:<14}  5x3 CV macro-F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
        print(classification_report(y_te, y_pred, zero_division=0))

        joblib.dump(model, f"{save_dir}/{name.lower()}_{target_col}.pkl")
        results[name] = (model, {
            "accuracy":          acc,
            "balanced_accuracy": bal_acc,
            "macro_f1":          mac_f1,
            "macro_precision":   mac_p,
            "macro_recall":      mac_r,
            "cv_macro_f1_mean":  round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std":   round(float(cv_f1.std()),  4),
            "y_test":            y_te,
            "y_pred":            y_pred,
        })

    return results
