"""K-Means, GMM, and Hierarchical clustering with K selection and PCA visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

PLOTS_DIR = "outputs/plots"
_CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


class _NearestCentroidWrapper:
    """Gives predict() to clusterers that only support fit_predict().

    After fit_predict(), each new point is assigned to the nearest
    cluster centroid (Euclidean distance in the original feature space).
    Used for AgglomerativeClustering which has no native predict().
    """

    def __init__(self, clusterer: Any, n_clusters: int) -> None:
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.centroids_: np.ndarray = np.array([])

    def fit_predict(self, X) -> np.ndarray:
        Xarr = X.values if hasattr(X, "values") else np.asarray(X)
        labels = self.clusterer.fit_predict(Xarr)
        self.centroids_ = np.array([
            Xarr[labels == k].mean(axis=0) if (labels == k).any() else Xarr.mean(axis=0)
            for k in range(self.n_clusters)
        ])
        return labels

    def predict(self, X) -> np.ndarray:
        Xarr = X.values if hasattr(X, "values") else np.asarray(X)
        dists = np.linalg.norm(Xarr[:, None] - self.centroids_[None, :], axis=2)
        return np.argmin(dists, axis=1)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature column names by excluding targets and cluster metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Fully-encoded and scaled DataFrame (post ``apply_encoding_and_scaling``).

    Returns
    -------
    list[str]
        Column names suitable for clustering and classification input.
    """
    exclude = {
        "stress_level",
        "burnout",
        "stress_level_encoded",
        "burnout_encoded",
        "cluster",
    }
    return [c for c in df.columns if c not in exclude]


def select_optimal_k(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_range: range = range(2, 7),
    save_dir: str = PLOTS_DIR,
) -> int:
    """Select optimal K for K-Means using silhouette score.

    Evaluates K in *k_range*, picks the K with the highest average silhouette
    score, and saves a two-panel PNG (inertia elbow + silhouette scores).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    k_range : range
        Candidate K values to evaluate (default 2..6).
    save_dir : str

    Returns
    -------
    int  Best K (highest silhouette score).
    """
    print(f"[clustering] Selecting optimal K via silhouette score {list(k_range)}...")
    X = df[feature_cols].values
    inertias: List[float] = []
    silhouettes: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            X, labels, sample_size=min(2000, len(X)), random_state=42
        )
        silhouettes.append(sil)
        print(f"[clustering]   K={k}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_idx = int(np.argmax(silhouettes))
    best_k = list(k_range)[best_idx]
    print(
        f"[clustering] Optimal K = {best_k}  "
        f"(silhouette={silhouettes[best_idx]:.4f})"
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, inertias, "bo-", markersize=8, linewidth=2)
    ax1.set_xlabel("K", fontsize=11)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=11)
    ax1.set_title("Elbow Curve", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, silhouettes, "rs-", markersize=8, linewidth=2)
    ax2.axvline(
        best_k, color="green", linestyle="--", alpha=0.8, label=f"Best K={best_k}"
    )
    ax2.set_xlabel("K", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score by K", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("K Selection: Elbow + Silhouette", fontsize=13)
    plt.tight_layout()
    out = f"{save_dir}/elbow_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[clustering] K-selection plot saved -> {out}")

    return best_k


def fit_kmeans(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3,
    save_dir: str = PLOTS_DIR,
) -> Tuple[pd.DataFrame, KMeans]:
    """Fit K-Means, label rows, and save a 2-D PCA cluster visualisation.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    n_clusters : int
        Number of clusters (default 3).
    save_dir : str

    Returns
    -------
    df : pd.DataFrame
        Copy of *df* with a ``cluster`` column added.
    kmeans : KMeans
        Fitted KMeans instance.
    """
    print(f"[clustering] Fitting K-Means with K={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(df[feature_cols])

    dist = df["cluster"].value_counts().sort_index()
    print(f"[clustering] Cluster distribution:\n{dist.to_string()}")

    # 2-D PCA visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(df[feature_cols])
    var1, var2 = pca.explained_variance_ratio_

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for cid in range(n_clusters):
        mask = df["cluster"].values == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=_CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)],
            label=f"Cluster {cid} (n={mask.sum()})",
            alpha=0.55,
            s=25,
        )
    ax.set_xlabel(f"PC1 ({var1:.1%} explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1%} explained)", fontsize=11)
    ax.set_title("K-Means Clusters — PCA 2-D Projection", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = f"{save_dir}/cluster_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[clustering] PCA visualisation saved -> {out}")

    return df, km


def fit_clustering_method(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    method: str = "kmeans",
) -> Tuple[pd.DataFrame, Any]:
    """Fit an alternative clustering method and return (df_with_cluster, clusterer).

    Parameters
    ----------
    method : "kmeans" | "gmm" | "hierarchical"
        - kmeans     : standard K-Means (same as fit_kmeans)
        - gmm        : Gaussian Mixture Model (full covariance)
        - hierarchical : Agglomerative Ward linkage + nearest-centroid predict

    Returns
    -------
    df : DataFrame with ``cluster`` column added.
    clusterer : fitted object with predict() method.
    """
    print(f"[clustering] Fitting {method.upper()} with K={n_clusters}...")
    X = df[feature_cols].values

    if method == "kmeans":
        clusterer: Any = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "gmm":
        clusterer = GaussianMixture(
            n_components=n_clusters, covariance_type="full",
            random_state=42, n_init=3,
        )
    elif method == "hierarchical":
        base = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        clusterer = _NearestCentroidWrapper(base, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}")

    df = df.copy()
    df["cluster"] = clusterer.fit_predict(X)

    dist = df["cluster"].value_counts().sort_index()
    print(f"[clustering] {method.upper()} cluster distribution:\n{dist.to_string()}")
    return df, clusterer
