"""
Villa Community Staffing Survey -- Analytics Dashboard (single-file version)
------------------------------------------------------------------------------
Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub (just app.py + requirements.txt +
               data/sample_data.csv -- no extra packages/folders to forget),
               then deploy on share.streamlit.io pointing at app.py.

Everything (data loading, feature engineering, descriptive/diagnostic stats,
classification, clustering) lives in this one file on purpose -- a previous
multi-file version (app.py importing from a separate utils/ package) caused a
ModuleNotFoundError on Streamlit Cloud when the utils/ folder didn't make it
into the GitHub repo correctly. A single file removes that failure mode.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, f_oneway
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_curve, auc)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Villa Staffing Survey Analytics", layout="wide")
RANDOM_STATE = 42

# =============================================================================
# 1. DATA LOADING + FEATURE ENGINEERING
# =============================================================================
TARGET_COL = "Q25_interest"
AGE_ORDER = {"Under 25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
INCOME_ORDER = {"Under 15,000": 0, "15,000-29,999": 1, "30,000-49,999": 2,
                 "50,000-79,999": 3, "80,000+": 4, "Unknown": -1}
NOMINAL_COLS = ["Q2_community", "Q3_nationality_region", "Q5_household_composition",
                "Q6_ownership", "Q11_sourcing_channel", "Q17_scheduling_pref",
                "Q20_billing_pref"]
CONTINUOUS_COLS = ["Q7_years_in_community", "Q14_hours_coordinating_monthly",
                    "Q18_current_spend_aed", "Q19_max_wtp_aed"]


def get_binary_cols(df):
    return [c for c in df.columns if c.startswith(("Q10_", "Q13_", "Q15_", "Q24_"))]


@st.cache_data
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)


def engineer_features(df, for_clustering=False):
    df = df.copy()
    df["Q9_income_bracket_aed"] = df["Q9_income_bracket_aed"].fillna("Unknown")
    df["income_ordinal"] = df["Q9_income_bracket_aed"].map(INCOME_ORDER)
    df["age_ordinal"] = df["Q4_age_group"].map(AGE_ORDER)
    for col in ["Q7_years_in_community", "Q14_hours_coordinating_monthly"]:
        df[col] = df[col].fillna(df[col].median())

    pain_cols = [c for c in df.columns if c.startswith("Q13_pain_") and c != "Q13_pain_none"]
    service_cols = [c for c in df.columns if c.startswith("Q10_") and c != "Q10_none"]
    feature_cols = [c for c in df.columns if c.startswith("Q15_feat_")]
    app_feat_cols = [c for c in df.columns if c.startswith("Q24_app_") and c != "Q24_app_none_prefer_whatsapp"]

    df["num_pain_points"] = df[pain_cols].sum(axis=1)
    df["num_services_used"] = df[service_cols].sum(axis=1)
    df["num_features_wanted"] = df[feature_cols].sum(axis=1)
    df["num_app_features_wanted"] = df[app_feat_cols].sum(axis=1)
    df["spend_to_wtp_gap"] = df["Q19_max_wtp_aed"] - df["Q18_current_spend_aed"]

    derived_continuous = ["num_pain_points", "num_services_used", "num_features_wanted",
                           "num_app_features_wanted", "spend_to_wtp_gap"]
    binary_cols = get_binary_cols(df)
    continuous_cols = CONTINUOUS_COLS + derived_continuous
    ordinal_cols = ["age_ordinal", "income_ordinal"]

    feature_cols_all = NOMINAL_COLS + continuous_cols + ordinal_cols + binary_cols
    X = df[feature_cols_all].copy()

    y = None
    if not for_clustering and TARGET_COL in df.columns:
        y = df[TARGET_COL].copy()

    meta = {
        "nominal_cols": NOMINAL_COLS,
        "continuous_cols": continuous_cols,
        "ordinal_cols": ordinal_cols,
        "binary_cols": binary_cols,
        "numeric_cols": continuous_cols + ordinal_cols,
    }
    return X, y, meta


def build_preprocessor(meta):
    return ColumnTransformer([
        ("numeric", StandardScaler(), meta["numeric_cols"]),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), meta["nominal_cols"]),
        ("binary", "passthrough", meta["binary_cols"]),
    ])


# =============================================================================
# 2. DESCRIPTIVE & DIAGNOSTIC ANALYSIS
# =============================================================================
def crosstab_pct(df, row_col, col_col=TARGET_COL, normalize="index"):
    counts = pd.crosstab(df[row_col], df[col_col])
    pct = pd.crosstab(df[row_col], df[col_col], normalize=normalize) * 100
    return counts, pct.round(1)


def cramers_v(confusion_matrix_):
    chi2 = chi2_contingency(confusion_matrix_)[0]
    n = confusion_matrix_.sum().sum()
    r, k = confusion_matrix_.shape
    phi2 = chi2 / n
    denom = min(k - 1, r - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan


def categorical_driver_ranking(df, target_col=TARGET_COL, candidate_cols=None):
    if candidate_cols is None:
        candidate_cols = NOMINAL_COLS + ["Q4_age_group", "Q9_income_bracket_aed",
                                          "Q17_scheduling_pref"] + get_binary_cols(df)
        candidate_cols = list(dict.fromkeys(candidate_cols))
    rows = []
    for col in candidate_cols:
        if col not in df.columns or col == target_col:
            continue
        ct = pd.crosstab(df[col], df[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        v = cramers_v(ct)
        rows.append({"feature": col, "chi2": round(chi2, 2), "p_value": round(p, 4),
                      "cramers_v": round(v, 3), "n": int(ct.sum().sum())})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_driver_ranking(df, target_col=TARGET_COL, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = CONTINUOUS_COLS + ["num_pain_points", "num_services_used",
                                           "num_features_wanted", "spend_to_wtp_gap"]
    rows = []
    groups_master = df[target_col].unique()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        groups = [df.loc[df[target_col] == g, col].dropna() for g in groups_master]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p = f_oneway(*groups)
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[col] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        rows.append({"feature": col, "F_stat": round(f_stat, 2), "p_value": round(p, 4),
                      "eta_squared": round(eta_sq, 3)})
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)


def correlation_matrix(df, cols=None):
    if cols is None:
        cols = CONTINUOUS_COLS + ["num_pain_points", "num_services_used", "num_features_wanted",
                                   "Q12_satisfaction", "Q16_continuity_importance",
                                   "Q22_vetting_importance", "Q23_app_comfort"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr().round(2)


def summary_statistics(df, cols=None):
    if cols is None:
        cols = CONTINUOUS_COLS
    return df[cols].describe().T.round(2)


# =============================================================================
# 3. CLASSIFICATION
# =============================================================================
MODEL_GRIDS = {
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 9, 15, 21],
                                      "clf__weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                       {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                       {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                           {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                            "clf__max_depth": [2, 3]}),
}


def prepare_train_test(df, test_size=0.25):
    X, y, meta = engineer_features(df)
    preprocessor = build_preprocessor(meta)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, preprocessor, meta


def train_all_models(df, test_size=0.25, cv=5, use_grid_search=True):
    X_train, X_test, y_train, y_test, preprocessor, meta = prepare_train_test(df, test_size)
    classes = sorted(y_train.unique())
    results = {}
    for name, (estimator, grid) in MODEL_GRIDS.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        if use_grid_search:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            best_pipe = pipe.fit(X_train, y_train)
            best_params = {}
        results[name] = {
            "pipeline": best_pipe, "best_params": best_params,
            "y_train": y_train, "y_train_pred": best_pipe.predict(X_train),
            "y_test": y_test, "y_test_pred": best_pipe.predict(X_test),
            "classes": classes,
        }
    return results, (X_train, X_test, y_train, y_test)


def attach_test_probabilities(results, X_test):
    for name, r in results.items():
        r["y_test_proba"] = r["pipeline"].predict_proba(X_test)
    return results


def metrics_table(results):
    rows = []
    for name, r in results.items():
        for split, y_true, y_pred in [("Train", r["y_train"], r["y_train_pred"]),
                                       ("Test", r["y_test"], r["y_test_pred"])]:
            rows.append({
                "Model": name, "Split": split,
                "Accuracy": round(accuracy_score(y_true, y_pred), 3),
                "Precision (weighted)": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 3),
                "Recall (weighted)": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 3),
                "F1 (weighted)": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 3),
            })
    return pd.DataFrame(rows)


def plot_metric_comparison(metrics_df):
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
    models = metrics_df["Model"].unique()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    width = 0.35
    x = np.arange(len(models))
    for ax, metric in zip(axes, metrics):
        train_vals = [metrics_df[(metrics_df.Model == m) & (metrics_df.Split == "Train")][metric].values[0] for m in models]
        test_vals = [metrics_df[(metrics_df.Model == m) & (metrics_df.Split == "Test")][metric].values[0] for m in models]
        ax.bar(x - width / 2, train_vals, width, label="Train", color="#4C72B0")
        ax.bar(x + width / 2, test_vals, width, label="Test", color="#DD8452")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrices(results):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    for ax, name in zip(axes, names):
        r = results[name]
        cm = confusion_matrix(r["y_test"], r["y_test_pred"], labels=r["classes"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=r["classes"], yticklabels=r["classes"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_roc_curves(results):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.5))
    colors = ["#2E8B57", "#E8A33D", "#C0392B"]
    for ax, name in zip(axes, names):
        r = results[name]
        classes = r["classes"]
        y_test_bin = label_binarize(r["y_test"], classes=classes)
        y_score = r.get("y_test_proba")
        if y_score is None:
            continue
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def classification_reports(results):
    reports = {}
    for name, r in results.items():
        rep = classification_report(r["y_test"], r["y_test_pred"], output_dict=True)
        reports[name] = pd.DataFrame(rep).T.round(3)
    return reports


# =============================================================================
# 4. CLUSTERING
# =============================================================================
def prepare_clustering_data(df):
    X, _, meta = engineer_features(df, for_clustering=True)
    preprocessor = build_preprocessor(meta)
    X_transformed = preprocessor.fit_transform(X)
    return X, X_transformed, meta


def elbow_and_silhouette(X_transformed, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_transformed)
        sil = silhouette_score(X_transformed, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": round(sil, 4)})
    return pd.DataFrame(rows)


def silhouette_best_k(elbow_df):
    best_score = elbow_df["silhouette_score"].max()
    candidates = elbow_df[elbow_df["silhouette_score"] >= best_score - 0.01]
    return int(candidates.sort_values("k").iloc[0]["k"])


def knee_point_k(elbow_df):
    x = elbow_df["k"].astype(float).values
    y = elbow_df["inertia"].astype(float).values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
    distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / \
        np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return int(x[np.argmax(distances)])


def recommend_k(elbow_df):
    return knee_point_k(elbow_df), silhouette_best_k(elbow_df)


def plot_elbow_chart(elbow_df, knee_k, sil_k):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="#4C72B0")
    axes[0].axvline(knee_k, color="#C0392B", linestyle="--", label=f"knee-point k={knee_k}")
    axes[0].set_title("Elbow chart (inertia vs. k)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[1].plot(elbow_df["k"], elbow_df["silhouette_score"], marker="o", color="#55A868")
    axes[1].axvline(sil_k, color="#C0392B", linestyle="--", label=f"silhouette-best k={sil_k}")
    if knee_k != sil_k:
        axes[1].axvline(knee_k, color="#4C72B0", linestyle=":", label=f"knee-point k={knee_k}")
    axes[1].set_title("Silhouette score vs. k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
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
    return coords, pca.explained_variance_ratio_


def plot_3d_clusters_plotly(coords, labels, explained, hover_df=None):
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
    profile_df = df.copy()
    profile_df["cluster"] = labels
    continuous_cols = ["Q7_years_in_community", "Q18_current_spend_aed", "Q19_max_wtp_aed"]
    for c in ["num_pain_points", "num_services_used", "num_features_wanted"]:
        if c in X.columns:
            profile_df[c] = X[c].values
            continuous_cols.append(c)
    numeric_profile = profile_df.groupby("cluster")[continuous_cols].mean().round(1)

    cat_profile = {}
    for col in ["Q6_ownership", "Q4_age_group", "Q25_interest"]:
        if col in profile_df.columns:
            cat_profile[col] = pd.crosstab(profile_df["cluster"], profile_df[col], normalize="index").round(2) * 100

    cluster_sizes = profile_df["cluster"].value_counts().sort_index()
    return numeric_profile, cat_profile, cluster_sizes, profile_df


def identify_best_cluster(numeric_profile, cat_profile, cluster_sizes):
    wtp = numeric_profile["Q19_max_wtp_aed"]
    wtp_norm = (wtp - wtp.min()) / (wtp.max() - wtp.min() + 1e-9)
    if "Q25_interest" in cat_profile and "Yes" in cat_profile["Q25_interest"].columns:
        yes_rate = cat_profile["Q25_interest"]["Yes"]
        yes_norm = (yes_rate - yes_rate.min()) / (yes_rate.max() - yes_rate.min() + 1e-9)
    else:
        yes_rate = pd.Series(np.nan, index=wtp_norm.index)
        yes_norm = pd.Series(0, index=wtp_norm.index)
    score = (wtp_norm + yes_norm) / 2
    score_df = pd.DataFrame({"avg_max_wtp_aed": wtp.round(0), "pct_yes_interest": yes_rate.round(1),
                              "cluster_size": cluster_sizes, "commercial_score": score.round(3)}
                             ).sort_values("commercial_score", ascending=False)
    return score_df.index[0], score_df


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.sidebar.title("Data source")
uploaded = st.sidebar.file_uploader("Upload survey CSV (optional)", type=["csv"])
if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success(f"Loaded {len(df)} rows from upload.")
else:
    default_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
    df = load_data(default_path)
    st.sidebar.info(f"Using bundled synthetic sample data ({len(df)} rows). "
                     "Upload your own CSV with the same column structure to replace it.")

st.title("🏘️ Villa Community Staffing Subscription — Survey Analytics")
st.caption("Descriptive & diagnostic analysis · classification · clustering — all computed live on the loaded dataset.")

tabs = st.tabs(["📊 Descriptive & Diagnostic", "🤖 Classification", "🧩 Clustering", "📝 Findings"])

# ---- TAB 1: Descriptive & Diagnostic ----
with tabs[0]:
    st.header("Descriptive analysis")
    st.markdown("*What does the data look like?*")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary statistics (numeric fields)")
        st.dataframe(summary_statistics(df), use_container_width=True)
    with col2:
        st.subheader("Correlation matrix")
        corr = correlation_matrix(df)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=False)
        st.pyplot(fig)

    st.subheader("Cross-tabulation explorer")
    cat_cols = ["Q4_age_group", "Q6_ownership", "Q5_household_composition", "Q11_sourcing_channel",
                "Q17_scheduling_pref", "Q20_billing_pref", "Q3_nationality_region"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    row_choice = st.selectbox("Cross-tabulate which variable against subscription interest (Q25)?", cat_cols)
    counts, pct = crosstab_pct(df, row_choice)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Counts**")
        st.dataframe(counts, use_container_width=True)
    with c2:
        st.markdown("**Row %**")
        st.dataframe(pct, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pct.reindex(columns=["No", "Maybe", "Yes"]).plot(kind="bar", stacked=True, ax=ax,
                                                       color=["#C0392B", "#E8A33D", "#2E8B57"])
    ax.set_ylabel("% of respondents")
    ax.legend(title="Interest", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

    st.divider()
    st.header("Diagnostic analysis")
    st.markdown("*Which factors are most associated with subscription interest, and how strongly?* "
                "With ~500 respondents, many relationships are statistically significant (low p-value) "
                "almost by default -- **effect size** (Cramer's V / eta-squared) is what tells you which "
                "ones actually matter.")

    df_eng, _, _ = engineer_features(df)
    df_diag = df.copy()
    for c in ["num_pain_points", "num_services_used", "num_features_wanted", "spend_to_wtp_gap"]:
        df_diag[c] = df_eng[c]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Categorical drivers (Chi-square + Cramer's V)")
        st.dataframe(categorical_driver_ranking(df_diag).head(12), use_container_width=True)
    with c2:
        st.subheader("Numeric drivers (ANOVA + eta-squared)")
        st.dataframe(numeric_driver_ranking(df_diag), use_container_width=True)

# ---- TAB 2: Classification ----
with tabs[1]:
    st.header("Classification: predicting subscription interest (Yes / No / Maybe)")
    st.markdown(
        "Four algorithms trained on the same engineered feature set: **KNN, Decision Tree, "
        "Random Forest, Gradient Boosting**. Each is tuned with a small grid search (5-fold CV), "
        "then evaluated on a held-out test set. Comparing **train vs. test** performance is the key "
        "stability check -- a large gap signals overfitting."
    )

    with st.expander("ℹ️ Feature engineering applied before training", expanded=False):
        st.markdown("""
- **Missing values:** income non-response kept as an explicit `Unknown` category; rare missing
  numeric fields median-imputed.
- **Ordinal encoding:** age group and income bracket mapped to ordered integers.
- **One-hot encoding:** nominal categoricals (community, nationality/region, household type,
  ownership, sourcing channel, scheduling preference, billing preference).
- **Binary flags:** all multi-select checklist items kept as 0/1 columns.
- **Derived features:** `num_pain_points`, `num_services_used`, `num_features_wanted`,
  `num_app_features_wanted`, `spend_to_wtp_gap`.
- **Scaling:** numeric/ordinal features standardized -- required for KNN, harmless for tree models.
- All fit only on the training split inside a scikit-learn `Pipeline` -- no leakage from test data.
        """)

    run_clf = st.button("▶ Train all 4 classifiers", type="primary")
    if run_clf or "clf_results" in st.session_state:
        if run_clf:
            with st.spinner("Training KNN, Decision Tree, Random Forest, Gradient Boosting (grid search)..."):
                results, (X_train, X_test, y_train, y_test) = train_all_models(df)
                results = attach_test_probabilities(results, X_test)
                st.session_state["clf_results"] = results
        results = st.session_state["clf_results"]

        st.subheader("Best hyperparameters found")
        st.json({name: r["best_params"] for name, r in results.items()})

        mt = metrics_table(results)
        st.subheader("Train vs. test accuracy / precision / recall / F1")
        st.dataframe(mt, use_container_width=True)
        st.pyplot(plot_metric_comparison(mt))

        st.subheader("Per-class detail (test set)")
        reports = classification_reports(results)
        model_choice = st.selectbox("Model", list(results.keys()), key="report_model")
        st.dataframe(reports[model_choice], use_container_width=True)

        st.subheader("Confusion matrices (test set)")
        st.pyplot(plot_confusion_matrices(results))

        st.subheader("ROC curves — one-vs-rest (test set)")
        st.markdown("A curve hugging the diagonal for a given class means the model can't reliably "
                     "separate that class from the rest -- watch the **'Maybe'** class especially.")
        st.pyplot(plot_roc_curves(results))
    else:
        st.info("Click the button above to train the models on the currently loaded data.")

# ---- TAB 3: Clustering ----
with tabs[2]:
    st.header("Clustering: discovering customer personas (K-means)")
    st.markdown("Unsupervised segmentation on demographics, behavior, and preferences "
                "(target/interest column excluded so clusters reflect *who they are*, not whether they said yes).")

    run_clust = st.button("▶ Run clustering analysis", type="primary")
    if run_clust or "clust_data" in st.session_state:
        if run_clust:
            with st.spinner("Fitting K-means for k=2..10 and computing silhouette scores..."):
                X, X_t, meta = prepare_clustering_data(df)
                elbow_df = elbow_and_silhouette(X_t)
                knee_k, sil_k = recommend_k(elbow_df)
                st.session_state["clust_data"] = dict(X=X, X_t=X_t, elbow_df=elbow_df, knee_k=knee_k, sil_k=sil_k)
        cd = st.session_state["clust_data"]
        X, X_t, elbow_df, knee_k, sil_k = cd["X"], cd["X_t"], cd["elbow_df"], cd["knee_k"], cd["sil_k"]

        st.subheader("Elbow chart & silhouette score")
        st.markdown(f"📈 **Knee-point method suggests k = {knee_k}**. "
                     f"📊 **Pure silhouette score is maximized at k = {sil_k}** (often too coarse for "
                     "designing multiple packages). Use the slider below to compare.")
        st.pyplot(plot_elbow_chart(elbow_df, knee_k, sil_k))
        st.dataframe(elbow_df, use_container_width=True)

        k_selected = st.slider("Number of clusters to use for the segmentation below",
                                min_value=2, max_value=10, value=knee_k)

        km, labels = fit_final_kmeans(X_t, k_selected)
        coords, explained = pca_3d_projection(X_t)
        st.caption(f"3D PCA projection explains {explained.sum()*100:.1f}% of total variance "
                    "(PC1 {:.1f}% + PC2 {:.1f}% + PC3 {:.1f}%) -- some overlap in the plot is expected.".format(
                        *[e * 100 for e in explained]))

        hover_cols = pd.DataFrame({"WTP_AED": df["Q19_max_wtp_aed"], "Interest": df["Q25_interest"]})
        st.plotly_chart(plot_3d_clusters_plotly(coords, labels, explained, hover_df=hover_cols),
                         use_container_width=True)

        numeric_profile, cat_profile, cluster_sizes, profile_df = profile_clusters(df, X, labels)
        st.subheader("Cluster profiles")
        st.dataframe(numeric_profile.assign(size=cluster_sizes), use_container_width=True)
        for col, table in cat_profile.items():
            with st.expander(f"{col} distribution by cluster (%)"):
                st.dataframe(table, use_container_width=True)

        best_cluster, score_df = identify_best_cluster(numeric_profile, cat_profile, cluster_sizes)
        st.subheader("🏆 Best cluster (most commercially attractive)")
        st.markdown("Defined as the cluster ranking highest on a combined score of "
                     "**willingness-to-pay** and **% saying Yes**.")
        st.dataframe(score_df, use_container_width=True)
        st.success(f"Cluster **{best_cluster}** looks most attractive: "
                    f"AED {score_df.loc[best_cluster, 'avg_max_wtp_aed']:.0f}/month avg willingness-to-pay, "
                    f"{score_df.loc[best_cluster, 'pct_yes_interest']:.0f}% saying Yes, "
                    f"{int(score_df.loc[best_cluster, 'cluster_size'])} respondents.")
    else:
        st.info("Click the button above to run K-means clustering on the currently loaded data.")

# ---- TAB 4: Findings ----
with tabs[3]:
    st.header("Findings")
    st.markdown("""
**From descriptive & diagnostic analysis:** check the Cramer's V / eta-squared tables, not just
p-values -- with a few hundred respondents, p-values are easily significant even for weak
relationships. Focus on whichever factors rank highest on effect size.

**From classification:** compare train vs. test metrics for every model. A model scoring ~1.0 on
train but much lower on test is overfitting. Check per-class ROC/AUC -- "Maybe" is usually hardest
to separate. Pick whichever model has the best *test* F1 with the smallest train-test gap.

**From clustering:** if silhouette score and the elbow knee-point disagree on k, that usually means
there's one dominant coarse split plus finer structure nested inside it -- try both k values with the
slider. Watch for a very small cluster (a handful of respondents) -- often outliers, not personas.

**Recommended next step:** once you have real pilot data, upload the new CSV in the sidebar and
re-check whether the same factors/clusters/model rankings hold up.
    """)
