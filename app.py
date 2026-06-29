"""
Villa Community Staffing Survey -- Analytics Dashboard
---------------------------------------------------------
Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on share.streamlit.io
               pointing at app.py (see README.md for full steps).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.data_processing import load_data, engineer_features
from utils.descriptive_diagnostic import (
    crosstab_pct, categorical_driver_ranking, numeric_driver_ranking,
    correlation_matrix, summary_statistics
)
from utils.classification_models import (
    train_all_models, metrics_table, plot_metric_comparison,
    plot_confusion_matrices, plot_roc_curves, attach_test_probabilities,
    classification_reports
)
from utils.clustering_analysis import (
    prepare_clustering_data, elbow_and_silhouette, recommend_k, plot_elbow_chart,
    fit_final_kmeans, pca_3d_projection, plot_3d_clusters_plotly,
    profile_clusters, identify_best_cluster
)

st.set_page_config(page_title="Villa Staffing Survey Analytics", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
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

# ===========================================================================
# TAB 1 — DESCRIPTIVE & DIAGNOSTIC
# ===========================================================================
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
        import seaborn as sns
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
    pct_plot = pct.reindex(columns=["No", "Maybe", "Yes"])
    pct_plot.plot(kind="bar", stacked=True, ax=ax, color=["#C0392B", "#E8A33D", "#2E8B57"])
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

# ===========================================================================
# TAB 2 — CLASSIFICATION
# ===========================================================================
with tabs[1]:
    st.header("Classification: predicting subscription interest (Yes / No / Maybe)")
    st.markdown(
        "Four algorithms are trained on the same engineered feature set: **KNN, Decision Tree, "
        "Random Forest, Gradient Boosting**. Each is tuned with a small grid search (5-fold CV), "
        "then evaluated on a held-out test set. Comparing **train vs. test** performance is the key "
        "stability check -- a large gap signals overfitting."
    )

    with st.expander("ℹ️ Feature engineering applied before training", expanded=False):
        st.markdown("""
- **Missing values:** income non-response kept as an explicit `Unknown` category (non-response is informative);
  rare missing numeric fields median-imputed.
- **Ordinal encoding:** age group and income bracket mapped to ordered integers (preserves order, unlike one-hot).
- **One-hot encoding:** nominal categoricals (community, nationality/region, household type, ownership, sourcing
  channel, scheduling preference, billing preference).
- **Binary flags:** all multi-select checklist items (services used, pain points, desired features, app features)
  kept as 0/1 columns.
- **Derived features:** `num_pain_points`, `num_services_used`, `num_features_wanted`, `num_app_features_wanted`,
  and `spend_to_wtp_gap` (willingness-to-pay minus current spend) -- aggregate signals that raw checklist columns
  don't capture individually.
- **Scaling:** all numeric/ordinal features standardized (mean 0, std 1) -- required for KNN's distance metric,
  harmless for the tree-based models.
- All of this is fit **only on the training split** inside a scikit-learn `Pipeline`, so there's no leakage from
  the test set into preprocessing.
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
                     "separate that class from the rest -- watch the **'Maybe'** class especially, "
                     "since it's the hardest to separate by nature.")
        st.pyplot(plot_roc_curves(results))
    else:
        st.info("Click the button above to train the models on the currently loaded data.")

# ===========================================================================
# TAB 3 — CLUSTERING
# ===========================================================================
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
        st.markdown(f"📈 **Knee-point method suggests k = {knee_k}** (best inertia/complexity trade-off). "
                     f"📊 **Pure silhouette score is maximized at k = {sil_k}** (the most cleanly separated split, "
                     "but often too coarse for designing multiple packages). These can legitimately disagree -- "
                     "use the slider below to compare.")
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
        fig3d = plot_3d_clusters_plotly(coords, labels, explained, hover_df=hover_cols)
        st.plotly_chart(fig3d, use_container_width=True)

        numeric_profile, cat_profile, cluster_sizes, profile_df = profile_clusters(df, X, labels)
        st.subheader("Cluster profiles")
        st.dataframe(numeric_profile.assign(size=cluster_sizes), use_container_width=True)
        for col, table in cat_profile.items():
            with st.expander(f"{col} distribution by cluster (%)"):
                st.dataframe(table, use_container_width=True)

        best_cluster, score_df = identify_best_cluster(numeric_profile, cat_profile, cluster_sizes)
        st.subheader("🏆 Best cluster (most commercially attractive)")
        st.markdown("Defined here as the cluster ranking highest on a combined score of "
                     "**willingness-to-pay** and **% saying Yes** -- adjust this definition in "
                     "`utils/clustering_analysis.py::identify_best_cluster` if your priorities differ.")
        st.dataframe(score_df, use_container_width=True)
        st.success(f"Cluster **{best_cluster}** looks most attractive: "
                    f"AED {score_df.loc[best_cluster, 'avg_max_wtp_aed']:.0f}/month avg willingness-to-pay, "
                    f"{score_df.loc[best_cluster, 'pct_yes_interest']:.0f}% saying Yes, "
                    f"{int(score_df.loc[best_cluster, 'cluster_size'])} respondents.")
    else:
        st.info("Click the button above to run K-means clustering on the currently loaded data.")

# ===========================================================================
# TAB 4 — FINDINGS
# ===========================================================================
with tabs[3]:
    st.header("Findings")
    st.markdown("""
This tab summarizes how to read the results from the other tabs. Numbers will update automatically
as you upload real survey data, so treat the specifics below as a *template for the kind of finding
to look for*, not a fixed conclusion.

**From descriptive & diagnostic analysis:**
- Check the Cramer's V / eta-squared tables, not just p-values -- with a few hundred respondents,
  p-values are easily significant even for weak relationships. Focus on whichever factors rank
  highest on effect size; those are your real segmentation/marketing levers.

**From classification:**
- Compare train vs. test metrics for every model. A model scoring ~1.0 on train but much lower on
  test (commonly true for KNN and Random Forest on small, noisy datasets) is overfitting and
  shouldn't be trusted for production scoring without more data or regularization (e.g. limiting
  tree depth, increasing K).
- Check the per-class ROC/AUC: the "Maybe" class is usually the hardest to separate (it's a genuinely
  ambiguous middle category) -- don't expect or require strong separation there.
- Whichever model has the best *test* (not train) F1 with the smallest train-test gap is the safer
  choice for actually scoring new respondents.

**From clustering:**
- If silhouette score and the elbow knee-point disagree on k (common with mixed categorical/numeric
  survey data), that's not a bug -- it usually means there's one dominant coarse split (e.g. a small
  high-value segment vs. everyone else) plus finer structure nested inside the larger group. Try both
  k values in the slider and see which produces cluster profiles that map onto packages you could
  actually sell.
- Watch for a very small cluster (a handful of respondents) -- that's often outliers/noisy responses
  being grouped together rather than a real persona. Worth inspecting those rows individually before
  building a package around them.
- The "best cluster" by commercial score isn't necessarily where to focus first -- a smaller,
  high-WTP cluster vs. a larger, more price-sensitive one is a strategic choice (premium tier vs.
  volume tier), not something the algorithm decides for you.

**Recommended next step once you have real pilot data:** re-run every tab as-is (just upload the new
CSV in the sidebar) and compare whether the same factors/clusters/model rankings hold up -- consistency
between the synthetic-data dry run and real data is itself a useful sanity check.
    """)
