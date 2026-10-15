"""All EDA and model-comparison visualisations — saves every plot as PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PLOTS_DIR = "outputs/plots"
_STRESS_PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def _save(filename: str, save_dir: str = PLOTS_DIR) -> None:
    """Save the current matplotlib figure and close it."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = f"{save_dir}/{filename}"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualization] Saved -> {out}")


# ── Plot 1 ──────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Lower-triangle correlation heatmap of all numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
        annot_kws={"size": 7},
    )
    plt.title("Correlation Heatmap — Numeric Features", fontsize=14)
    plt.tight_layout()
    _save("correlation_heatmap.png", save_dir)


# ── Plot 2 ──────────────────────────────────────────────────────────────────

def plot_distribution_histograms(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """2×2 grid of distribution histograms for the four primary lifestyle features."""
    priority = [
        "anxiety_level", "depression", "self_esteem", "sleep_quality",
        "study_load", "social_support",
        "sleep_hours", "study_hours", "social_hours", "physical_activity",
        "extracurricular_hours", "academic_performance",
    ]
    cols = [c for c in priority if c in df.columns][:6]
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=35, color="steelblue", edgecolor="white", alpha=0.85)
        axes[i].set_title(f"Distribution of {col}", fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Lifestyle Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    _save("distribution_histograms.png", save_dir)


# ── Plot 3 ──────────────────────────────────────────────────────────────────

def plot_stress_distribution(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Annotated bar chart of stress-level class counts."""
    if "stress_level" not in df.columns:
        print("[visualization] stress_level column not found - skipping.")
        return
    order = [v for v in ["Low", "Medium", "High"] if v in df["stress_level"].values]
    counts = df["stress_level"].value_counts().reindex(order)
    colors = [_STRESS_PALETTE[o] for o in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Stress Level Class Distribution", fontsize=13)
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(val),
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    _save("stress_distribution.png", save_dir)


# ── Plot 4 ──────────────────────────────────────────────────────────────────

def plot_burnout_distribution(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Pie chart of burnout (Yes/No) proportions."""
    if "burnout" not in df.columns:
        print("[visualization] burnout column not found - skipping.")
        return
    counts = df["burnout"].value_counts()
    colors = ["#e74c3c" if k == "Yes" else "#2ecc71" for k in counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("Burnout Distribution", fontsize=13)
    plt.tight_layout()
    _save("burnout_distribution.png", save_dir)


# ── Plot 5 ──────────────────────────────────────────────────────────────────

def plot_boxplots_by_stress(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Boxplots of each numeric feature split by stress level."""
    if "stress_level" not in df.columns:
        return
    priority = [
        "anxiety_level", "depression", "self_esteem", "sleep_quality",
        "study_load", "social_support",
        "sleep_hours", "study_hours", "social_hours", "physical_activity",
        "extracurricular_hours", "academic_performance",
    ]
    numeric_cols = [c for c in priority if c in df.columns][:6]
    order = [v for v in ["Low", "Medium", "High"] if v in df["stress_level"].values]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        groups = [df[df["stress_level"] == lvl][col].dropna() for lvl in order]
        bp = axes[i].boxplot(groups, tick_labels=order, patch_artist=True)
        for patch, lvl in zip(bp["boxes"], order):
            patch.set_facecolor(_STRESS_PALETTE.get(lvl, "#aaa"))
            patch.set_alpha(0.7)
        axes[i].set_title(f"{col} by Stress Level", fontsize=10)
        axes[i].set_xlabel("Stress Level")
        axes[i].set_ylabel(col)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Stress Level", fontsize=14)
    plt.tight_layout()
    _save("boxplots_by_stress.png", save_dir)


# ── Plot 6 ──────────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Pairplot of key features coloured by stress_level (sampled for speed)."""
    if "stress_level" not in df.columns:
        return
    pairplot_priority = [
        "anxiety_level", "depression", "sleep_quality", "study_load",
        "social_support", "self_esteem",
        "sleep_hours", "study_hours", "social_hours", "physical_activity",
    ]
    keep = [c for c in pairplot_priority if c in df.columns][:4] + ["stress_level"]
    sample = df[keep].sample(min(600, len(df)), random_state=42)

    g = sns.pairplot(
        sample,
        hue="stress_level",
        palette=_STRESS_PALETTE,
        hue_order=["Low", "Medium", "High"],
        plot_kws={"alpha": 0.45, "s": 18},
        diag_kind="kde",
    )
    g.fig.suptitle("Pairplot of Key Features by Stress Level", y=1.02, fontsize=13)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = f"{save_dir}/pairplot.png"
    g.fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"[visualization] Saved -> {out}")


# ── Plot 7 ──────────────────────────────────────────────────────────────────

def plot_feature_importance_bar(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Global RF Feature Importance",
    save_dir: str = PLOTS_DIR,
) -> None:
    """Standalone horizontal feature-importance bar chart saved to plots dir."""
    idx = np.argsort(importances)
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.35)))
    ax.barh(range(len(names)), vals, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    _save("feature_importance.png", save_dir)


# ── Plot 11 ─────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_dict: Dict[str, Dict],
    save_dir: str = PLOTS_DIR,
    n_test: int = 220,
) -> None:
    """Grouped bar chart for the global model comparison (Hybrid vs Baseline).

    Both models are evaluated on the same held-out test set so the comparison
    is fair.  A chance-level reference line (1/3 for 3 classes) is included.
    """
    models = list(results_dict.keys())
    accuracies = [results_dict[m].get("accuracy", 0) for m in models]
    macro_f1s = [results_dict[m].get("macro_f1", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.2), 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, macro_f1s, width, label="Macro F1",
                   color="coral", alpha=0.85)

    ax.axhline(1 / 3, color="gray", linestyle="--", alpha=0.55,
               label="Chance level (0.33)")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Global Model Comparison\n(evaluated on {n_test} held-out test samples)",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save("model_comparison.png", save_dir)


# ── Plot 12 ─────────────────────────────────────────────────────────────────

def plot_cluster_diagnostics(
    df_train: pd.DataFrame,
    stress_results: Dict,
    save_dir: str = PLOTS_DIR,
) -> None:
    """Two-panel cluster diagnostics: stress composition + best per-cluster F1.

    Left panel shows how many Low / Medium / High students fall in each
    cluster (training set).  Right panel shows the best model's macro-F1
    per cluster with a note that these are *internal* evaluations, not
    the global test set.
    """
    clusters = sorted(df_train["cluster"].unique())
    n = len(clusters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: stacked composition ────────────────────────────────────────────
    stress_order = ["Low", "Medium", "High"]
    comp_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    bottoms = np.zeros(n)
    for lvl in stress_order:
        counts = []
        for cid in clusters:
            sub = df_train[df_train["cluster"] == cid]
            counts.append((sub.get("stress_level", pd.Series()) == lvl).sum())
        ax1.bar(range(n), counts, bottom=bottoms, label=lvl,
                color=comp_colors[lvl], alpha=0.85, edgecolor="white")
        bottoms += np.array(counts, dtype=float)

    ax1.set_xlabel("Cluster", fontsize=11)
    ax1.set_ylabel("Number of Students (Training Set)", fontsize=11)
    ax1.set_title("Stress Class Composition per Cluster\n(Training Set)", fontsize=12)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([f"C{c}" for c in clusters])
    ax1.legend(title="Stress Level", fontsize=9)

    # ── Right: best per-cluster macro-F1 ────────────────────────────────────
    best_labels, best_f1s, bar_colors = [], [], []
    for cid in clusters:
        res = stress_results.get(cid, {})
        if res:
            best_name = max(res, key=lambda nm: res[nm]["macro_f1"])
            f1 = res[best_name]["macro_f1"]
            display = "Trivial" if best_name == "DummyClassifier" else best_name[:3]
            col = "#95a5a6" if best_name == "DummyClassifier" else (
                "#2ecc71" if f1 >= 0.7 else "#e67e22" if f1 >= 0.4 else "#e74c3c"
            )
        else:
            display, f1, col = "None", 0.0, "#aaa"
        best_labels.append(f"C{cid}\n({display})")
        best_f1s.append(f1)
        bar_colors.append(col)

    ax2.bar(range(n), best_f1s, color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.axhline(1 / 3, color="gray", linestyle="--", alpha=0.55, label="Chance (0.33)")
    ax2.set_xlabel("Cluster", fontsize=11)
    ax2.set_ylabel("Best Macro-F1", fontsize=11)
    ax2.set_title(
        "Best Per-Cluster Model Performance\n(Internal Cluster Eval -- NOT Global Test Set)",
        fontsize=12,
    )
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(best_labels, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=9)
    for i, f1 in enumerate(best_f1s):
        ax2.text(i, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=9)

    plt.suptitle("Cluster Diagnostics: Composition and Internal Performance", fontsize=13)
    plt.tight_layout()
    _save("cluster_diagnostics.png", save_dir)


# ── Plot 13 ─────────────────────────────────────────────────────────────────

def plot_importance_vs_correlation(
    importances: np.ndarray,
    feature_names: List[str],
    df_train: pd.DataFrame,
    target_col: str = "stress_level_encoded",
    save_dir: str = PLOTS_DIR,
) -> None:
    """Scatter: absolute Pearson correlation (EDA) vs RF importance (model).

    If both methods agree on the top features, the points form a diagonal
    trend — confirming EDA findings are reflected in the learned model.
    """
    corrs = np.array([
        abs(df_train[f].corr(df_train[target_col])) if f in df_train.columns else 0.0
        for f in feature_names
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(corrs, importances, s=60, alpha=0.75, color="steelblue", edgecolors="white")

    for i, name in enumerate(feature_names):
        ax.annotate(
            name, (corrs[i], importances[i]),
            fontsize=7, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points",
        )

    if len(corrs) > 1:
        z = np.polyfit(corrs, importances, 1)
        xline = np.linspace(corrs.min(), corrs.max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), "r--", alpha=0.5, label="trend")
        ax.legend(fontsize=9)

    ax.set_xlabel("|Pearson correlation with stress_level|", fontsize=11)
    ax.set_ylabel("RF Feature Importance", fontsize=11)
    ax.set_title(
        "EDA Correlation vs RF Feature Importance\n"
        "(diagonal trend = EDA findings confirmed by the model)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("importance_vs_correlation.png", save_dir)


# ── Plot 14 — Cluster feature heatmap ────────────────────────────────────────

def plot_cluster_feature_heatmap(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    save_dir: str = PLOTS_DIR,
) -> None:
    """Heatmap of per-cluster feature deviations from the global mean.

    Each cell shows  cluster_mean(feature) - global_mean(feature)  in the
    [0,1]-scaled space.  Red = this cluster scores higher than average on this
    feature; Blue = lower.  Rows are labelled with the cluster's dominant
    stress class so the reader can immediately connect the feature pattern to
    the stress outcome.
    """
    if "cluster" not in df_train.columns:
        return
    clusters = sorted(df_train["cluster"].unique())
    global_mean = df_train[feature_cols].mean()

    rows, row_labels = [], []
    for cid in clusters:
        sub = df_train[df_train["cluster"] == cid]
        rows.append((sub[feature_cols].mean() - global_mean).values)
        dom = sub["stress_level"].mode().iloc[0] if "stress_level" in sub.columns else "?"
        n = len(sub)
        row_labels.append(f"C{cid}  [{dom} stress, n={n}]")

    dev_df = pd.DataFrame(rows, index=row_labels, columns=feature_cols)

    fig, ax = plt.subplots(figsize=(18, max(3, len(clusters) * 1.2 + 1)))
    sns.heatmap(
        dev_df,
        annot=False,
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Deviation from global mean (MinMax-scaled)"},
    )
    ax.set_title(
        "Cluster Feature Profiles — Deviation from Global Mean\n"
        "(Red = above avg, Blue = below avg; features in [0,1] MinMax scale)",
        fontsize=12,
    )
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _save("cluster_feature_heatmap.png", save_dir)


# ── Plot 15 — Cluster radar chart ─────────────────────────────────────────────

def plot_cluster_radar(
    df_train: pd.DataFrame,
    save_dir: str = PLOTS_DIR,
) -> None:
    """Radar / spider chart comparing clusters on the 5 composite features.

    Uses only the engineered composite features (which are interpretable in
    psychological terms) rather than all 25 features, keeping the chart readable.
    """
    if "cluster" not in df_train.columns:
        return

    composite_cols = [
        "psychological_burden", "academic_stress", "wellbeing_index",
        "environmental_stress", "support_deficit",
    ]
    cols = [c for c in composite_cols if c in df_train.columns]
    if len(cols) < 3:
        return

    clusters = sorted(df_train["cluster"].unique())
    n_ax = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    _colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
    stress_label_map: dict = {}
    if "stress_level" in df_train.columns:
        for cid in clusters:
            sub = df_train[df_train["cluster"] == cid]
            dom = sub["stress_level"].mode().iloc[0] if len(sub) else "?"
            stress_label_map[cid] = dom

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, cid in enumerate(clusters):
        sub = df_train[df_train["cluster"] == cid]
        vals = [sub[c].mean() for c in cols]
        vals += vals[:1]
        dom = stress_label_map.get(cid, "?")
        label = f"C{cid}  ({dom} stress, n={len(sub)})"
        color = _colors[i % len(_colors)]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title(
        "Cluster Profiles — Composite Features\n"
        "(MinMax-scaled means; higher = more of that construct)",
        fontsize=12, pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=9)
    plt.tight_layout()
    _save("cluster_radar.png", save_dir)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def run_all_eda(df: pd.DataFrame, save_dir: str = PLOTS_DIR) -> None:
    """Run all EDA plots (1–6) in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-scaled DataFrame that still contains the original target string
        columns (``stress_level``, ``burnout``) and numeric feature columns.
    save_dir : str
        Output directory for PNGs.
    """
    print("\n[visualization] Generating EDA plots...")
    plot_correlation_heatmap(df, save_dir)
    plot_distribution_histograms(df, save_dir)
    plot_stress_distribution(df, save_dir)
    plot_burnout_distribution(df, save_dir)
    plot_boxplots_by_stress(df, save_dir)
    plot_pairplot(df, save_dir)
    print("[visualization] EDA plots complete.\n")
