"""Evaluation utilities: confusion matrix, feature importance, CSV report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import classification_report, confusion_matrix, f1_score


PLOTS_DIR = "outputs/plots"
RESULTS_DIR = "outputs/results"


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    title: str = "Confusion Matrix",
    save_dir: str = PLOTS_DIR,
) -> None:
    """Plot a labelled heatmap confusion matrix and save as PNG.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    title : str
        Also used to derive the output filename.
    save_dir : str
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar=False,
    )
    plt.title(title, fontsize=12)
    plt.ylabel("True Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()

    safe_name = (
        title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )
    out = f"{save_dir}/{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluation] Confusion matrix saved -> {out}")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    title: str = "Feature Importance",
    save_dir: str = PLOTS_DIR,
    top_n: int = 20,
) -> None:
    """Plot and save a horizontal bar chart of feature importances.

    Only works for models that expose ``feature_importances_`` (tree-based).

    Parameters
    ----------
    model : fitted sklearn / xgboost model
    feature_names : list[str]
    title : str
    save_dir : str
    top_n : int
        Maximum number of features to display.
    """
    if not hasattr(model, "feature_importances_"):
        print("[evaluation] Model has no feature_importances_ - skipping plot.")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(range(len(names))[::-1], vals, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    safe_name = title.lower().replace(" ", "_")
    out = f"{save_dir}/{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluation] Feature importance plot saved -> {out}")


def generate_report(
    results_dict: Dict[str, Dict],
    save_dir: str = RESULTS_DIR,
) -> pd.DataFrame:
    """Compile a model-comparison table and save to CSV.

    Parameters
    ----------
    results_dict : dict
        ``{model_label: {"accuracy": float, "macro_f1": float, ...}}``.
    save_dir : str

    Returns
    -------
    pd.DataFrame
        Sorted by macro_f1 descending.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": label,
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
            "balanced_accuracy": round(metrics.get("balanced_accuracy", 0.0), 4),
            "macro_precision": round(metrics.get("macro_precision", 0.0), 4),
            "macro_recall": round(metrics.get("macro_recall", 0.0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
            "f1_ci_lower": round(metrics.get("f1_ci_lower", 0.0), 4),
            "f1_ci_upper": round(metrics.get("f1_ci_upper", 0.0), 4),
            "cv_macro_f1_mean": round(metrics.get("cv_macro_f1_mean", 0.0), 4),
            "cv_macro_f1_std": round(metrics.get("cv_macro_f1_std", 0.0), 4),
        }
        for label, metrics in results_dict.items()
    ]
    report_df = (
        pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    )
    out = f"{save_dir}/model_comparison.csv"
    report_df.to_csv(out, index=False)
    print(f"[evaluation] Report saved -> {out}")
    return report_df


def bootstrap_ci(
    y_true: Any,
    y_pred: Any,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """Percentile bootstrap confidence interval for macro-F1.

    Resamples (y_true, y_pred) pairs *with replacement* n_boot times and
    computes macro-F1 on each resample.  Returns the alpha/2 and
    1-alpha/2 percentiles as (lower, upper).

    Parameters
    ----------
    y_true, y_pred : array-like
    n_boot : int   Number of bootstrap resamples (default 1000).
    alpha  : float Significance level (default 0.05 -> 95% CI).
    random_state : int

    Returns
    -------
    (lower, upper) : tuple[float, float]
    """
    rng = np.random.RandomState(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)
    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_scores[i] = f1_score(
            y_true_arr[idx], y_pred_arr[idx], average="macro", zero_division=0
        )
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return lower, upper


def print_classification_report(y_true: Any, y_pred: Any) -> None:
    """Print a formatted sklearn classification report."""
    print(classification_report(y_true, y_pred, zero_division=0))


def mcnemar_test(
    y_true: Any,
    y_pred_1: Any,
    y_pred_2: Any,
    model1_name: str = "Model1",
    model2_name: str = "Model2",
) -> float:
    """McNemar's test for statistical significance between two classifiers.

    Uses Yates' continuity correction:
        chi2 = (|b - c| - 1)^2 / (b + c)
    where
        b = samples correct by model_2 but NOT model_1
        c = samples correct by model_1 but NOT model_2

    Returns
    -------
    float  p-value (< 0.05 => significant difference at 95% confidence)
    """
    y_true = np.array(y_true)
    y1 = np.array(y_pred_1)
    y2 = np.array(y_pred_2)

    correct_1 = y1 == y_true
    correct_2 = y2 == y_true

    b = int(np.sum(~correct_1 & correct_2))
    c = int(np.sum(correct_1 & ~correct_2))

    if b + c == 0:
        print(
            f"[evaluation] McNemar ({model1_name} vs {model2_name}): "
            "models agree on every sample  p=1.000"
        )
        return 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
    print(
        f"[evaluation] McNemar  {model1_name} vs {model2_name}: "
        f"b={b}  c={c}  chi2={statistic:.3f}  p={p_value:.4f}  -> {sig}"
    )
    return p_value
