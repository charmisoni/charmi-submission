"""
descriptive_diagnostic.py
--------------------------
Descriptive analysis  -> "what does the data look like" (distributions, cross-tabs)
Diagnostic analysis    -> "why/what's associated with interest" (significance tests,
                           effect sizes, driver ranking)

Framework-agnostic: returns pandas DataFrames/dicts that the Streamlit app
(or a notebook) renders however it likes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

from .data_processing import TARGET_COL, NOMINAL_COLS, CONTINUOUS_COLS, get_binary_cols


def crosstab_pct(df, row_col, col_col=TARGET_COL, normalize="index"):
    """Descriptive: count + % cross-tabulation between two categorical columns."""
    counts = pd.crosstab(df[row_col], df[col_col])
    pct = pd.crosstab(df[row_col], df[col_col], normalize=normalize) * 100
    return counts, pct.round(1)


def cramers_v(confusion_matrix):
    """Effect size for chi-square test of independence (0 = no association, 1 = perfect)."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    denom = min(k - 1, r - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan


def categorical_driver_ranking(df, target_col=TARGET_COL, candidate_cols=None):
    """
    Diagnostic analysis: for every categorical predictor, run a chi-square test
    of independence against the target and rank by Cramer's V (effect size),
    not just p-value -- with n=500 almost everything is "significant", so
    effect size is what actually tells you which factors matter most.
    """
    if candidate_cols is None:
        candidate_cols = NOMINAL_COLS + ["Q4_age_group", "Q9_income_bracket_aed",
                                          "Q17_scheduling_pref"] + get_binary_cols(df)
        candidate_cols = list(dict.fromkeys(candidate_cols))  # de-dup, keep order

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
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return result


def numeric_driver_ranking(df, target_col=TARGET_COL, numeric_cols=None):
    """
    Diagnostic analysis: one-way ANOVA for each numeric feature across the
    target groups, with eta-squared as the effect size (analogous role to
    Cramer's V but for continuous variables).
    """
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
        # eta squared = SS_between / SS_total
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[col] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        rows.append({"feature": col, "F_stat": round(f_stat, 2), "p_value": round(p, 4),
                      "eta_squared": round(eta_sq, 3)})
    result = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    return result


def correlation_matrix(df, cols=None):
    """Descriptive: Pearson correlation among numeric/derived features."""
    if cols is None:
        cols = CONTINUOUS_COLS + ["num_pain_points", "num_services_used",
                                   "num_features_wanted", "Q12_satisfaction",
                                   "Q16_continuity_importance", "Q22_vetting_importance",
                                   "Q23_app_comfort"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr().round(2)


def summary_statistics(df, cols=None):
    """Descriptive: standard summary stats table for numeric columns."""
    if cols is None:
        cols = CONTINUOUS_COLS
    return df[cols].describe().T.round(2)
