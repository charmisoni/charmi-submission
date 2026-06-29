"""
clustering_analysis.py
------------------------
Unsupervised segmentation via K-means:
  - elbow chart (inertia vs k) + silhouette score (a second, more objective
    signal for picking k, since "elbow" by eye is subjective)
  - PCA to 3 components for a 3D visualization of the resulting clusters
  - cluster profiling (what makes each cluster distinct)
  - "best cluster" = the most commercially attractive segment, defined here
    as the cluster with the highest combined rank of (a) willingness to pay
    and (b) subscription interest rate -- this definition is a business
    judgment call, not a statistical one, and is stated explicitly so you
    can swap in a different definition if yours differs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .data_processing import engineer_features, build_preprocessor

RANDOM_STATE = 42


def prepare_clustering_data(df):
    X, _, meta = engineer_features(df, for_clustering=True)
    preprocessor = build_preprocessor(meta)
    X_transformed = preprocessor.fit_transform(X)
    return X, X_transformed, meta


def elbow_and_silhouette(X_transformed, k_range=range(2, 11)):
    """Fit K-means for every k in k_range; return inertia + silhouette per k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_transformed)
        sil = silhouette_score(X_transformed, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": round(sil, 4)})
    return pd.DataFrame(rows)


def silhouette_best_k(elbow_df):
    """Pick k by pure silhouette score (tends to favor coarse, well-separated splits)."""
    best_score = elbow_df["silhouette_score"].max()
    candidates = elbow_df[elbow_df["silhouette_score"] >= best_score - 0.01]
    return int(candidates.sort_values("k").iloc[0]["k"])


def knee_point_k(elbow_df):
    """
    Classic 'kneedle' method: normalize the inertia curve to [0,1] on both
    axes, draw a straight line from the first to the last point, and pick the
    k with the maximum perpendicular distance from that line. This usually
    aligns better with a *business-meaningful* number of segments than
    silhouette score alone, which on high-dimensional one-hot survey data
    tends to just find the single coarsest, most-separated split.
    """
    x = elbow_df["k"].astype(float).values
    y = elbow_df["inertia"].astype(float).values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
    distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / \
        np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return int(x[np.argmax(distances)])


def recommend_k(elbow_df):
    """
    Returns the knee-point k as the primary recommendation (better suited to
    multi-segment business use), alongside the pure silhouette-best k for
    transparency -- the two can legitimately disagree, and that disagreement
    is itself worth reporting rather than hiding.
    """
    knee_k = knee_point_k(elbow_df)
    sil_k = silhouette_best_k(elbow_df)
    return knee_k, sil_k


def plot_elbow_chart(elbow_df, knee_k, sil_k):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="#4C72B0")
    axes[0].axvline(knee_k, color="#C0392B", linestyle="--", label=f"knee-point k={knee_k}")
    axes[0].set_title("Elbow chart (inertia vs. k)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[0].legend()

    axes[1].plot(elbow_df["k"], elbow_df["silhouette_score"], marker="o", color="#55A868")
    axes[1].axvline(sil_k, color="#C0392B", linestyle="--", label=f"silhouette-best k={sil_k}")
    if knee_k != sil_k:
        axes[1].axvline(knee_k, color="#4C72B0", linestyle=":", label=f"knee-point k={knee_k}")
    axes[1].set_title("Silhouette score vs. k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score (higher = better-separated clusters)")
    axes[1].legend()
    plt.tight_layout()
    return fig


def fit_final_kmeans(X_transformed, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_transformed)
    return km, labels


def pca_3d_projection(X_transformed):
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_transformed)
    explained = pca.explained_variance_ratio_
    return coords, explained


def plot_3d_clusters_matplotlib(coords, labels, explained, title="K-means clusters (PCA projection)"):
    """Static 3D plot -- this is the version meant for the Colab script."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c=labels, cmap="Set2", s=40, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}% var)")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
    ax.add_artist(legend)
    plt.tight_layout()
    return fig


def plot_3d_clusters_plotly(coords, labels, explained, hover_df=None):
    """Interactive 3D plot -- this is the version used inside the Streamlit app."""
    import plotly.express as px
    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    plot_df["Cluster"] = labels.astype(str)
    if hover_df is not None:
        for col in hover_df.columns:
            plot_df[col] = hover_df[col].values

    fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color="Cluster",
        hover_data=[c for c in (hover_df.columns if hover_df is not None else [])],
        opacity=0.8,
        title=f"K-means clusters in 3D PCA space "
              f"(PC1 {explained[0]*100:.1f}% + PC2 {explained[1]*100:.1f}% + PC3 {explained[2]*100:.1f}% var explained)"
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="DarkSlateGrey")))
    return fig


def profile_clusters(df, X, labels):
    """
    Mean of key continuous features + rate of key binary features per cluster,
    so each cluster can be described in plain business language.
    """
    profile_df = df.copy()
    profile_df["cluster"] = labels

    continuous_cols = ["Q7_years_in_community", "Q18_current_spend_aed", "Q19_max_wtp_aed",
                        "num_pain_points" if "num_pain_points" in X.columns else None,
                        "num_services_used" if "num_services_used" in X.columns else None]
    continuous_cols = [c for c in continuous_cols if c is not None]
    for c in ["num_pain_points", "num_services_used", "num_features_wanted"]:
        if c in X.columns:
            profile_df[c] = X[c].values

    numeric_profile = profile_df.groupby("cluster")[continuous_cols].mean().round(1)

    cat_profile = {}
    for col in ["Q6_ownership", "Q4_age_group", "Q25_interest"]:
        if col in profile_df.columns:
            cat_profile[col] = pd.crosstab(profile_df["cluster"], profile_df[col], normalize="index").round(2) * 100

    cluster_sizes = profile_df["cluster"].value_counts().sort_index()
    return numeric_profile, cat_profile, cluster_sizes, profile_df


def identify_best_cluster(numeric_profile, cat_profile, cluster_sizes):
    """
    Rank clusters by a composite 'commercial attractiveness' score:
    normalized willingness-to-pay + normalized % 'Yes' interest.
    Returns the best cluster id and the score table (so the choice is auditable).
    """
    wtp = numeric_profile["Q19_max_wtp_aed"]
    wtp_norm = (wtp - wtp.min()) / (wtp.max() - wtp.min() + 1e-9)

    if "Q25_interest" in cat_profile and "Yes" in cat_profile["Q25_interest"].columns:
        yes_rate = cat_profile["Q25_interest"]["Yes"]
        yes_norm = (yes_rate - yes_rate.min()) / (yes_rate.max() - yes_rate.min() + 1e-9)
    else:
        yes_norm = pd.Series(0, index=wtp_norm.index)

    score = (wtp_norm + yes_norm) / 2
    score_df = pd.DataFrame({"avg_max_wtp_aed": wtp.round(0), "pct_yes_interest": yes_rate.round(1)
                              if "Q25_interest" in cat_profile else np.nan,
                              "cluster_size": cluster_sizes,
                              "commercial_score": score.round(3)}).sort_values("commercial_score", ascending=False)
    best_cluster = score_df.index[0]
    return best_cluster, score_df
