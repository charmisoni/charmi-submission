"""
data_processing.py
-------------------
Loading + feature engineering for the villa community staffing survey data.

This module is intentionally framework-agnostic (no Streamlit imports) so it
can be unit-tested directly, reused in a notebook/Colab, or imported by the
Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ["respondent_id", "_true_persona"]  # _true_persona is ground truth, never a model input
TARGET_COL = "Q25_interest"

AGE_ORDER = {"Under 25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
INCOME_ORDER = {"Under 15,000": 0, "15,000-29,999": 1, "30,000-49,999": 2,
                 "50,000-79,999": 3, "80,000+": 4}

NOMINAL_COLS = ["Q2_community", "Q3_nationality_region", "Q5_household_composition",
                "Q6_ownership", "Q11_sourcing_channel", "Q17_scheduling_pref",
                "Q20_billing_pref"]

CONTINUOUS_COLS = ["Q7_years_in_community", "Q14_hours_coordinating_monthly",
                    "Q18_current_spend_aed", "Q19_max_wtp_aed"]

# binary multi-select columns already encoded as 0/1 in the raw data
def get_binary_cols(df):
    return [c for c in df.columns if c.startswith(("Q10_", "Q13_", "Q15_", "Q24_"))]


def load_data(path_or_buffer):
    """Load the raw survey CSV (path string or an uploaded file-like object)."""
    df = pd.read_csv(path_or_buffer)
    return df


def engineer_features(df, for_clustering=False):
    """
    Apply consistent feature engineering to the raw survey dataframe.

    Returns
    -------
    X : pd.DataFrame   engineered feature matrix (NOT yet encoded/scaled)
    y : pd.Series or None   target labels (None if for_clustering=True or target absent)
    meta : dict          column-role info needed to build the ColumnTransformer
    """
    df = df.copy()

    # --- 1. Missing value handling -----------------------------------------
    # Income: non-response is itself informative (often correlates with high-
    # or low-income respondents declining), so we keep it as an explicit
    # "Unknown" category rather than imputing a numeric value.
    df["Q9_income_bracket_aed"] = df["Q9_income_bracket_aed"].fillna("Unknown")
    income_order_with_unknown = {**INCOME_ORDER, "Unknown": -1}
    df["income_ordinal"] = df["Q9_income_bracket_aed"].map(income_order_with_unknown)

    # Years in community / hours coordinating: rare item non-response,
    # median imputation is a reasonable, low-distortion default at <2% missing.
    for col in ["Q7_years_in_community", "Q14_hours_coordinating_monthly"]:
        df[col] = df[col].fillna(df[col].median())

    # --- 2. Ordinal encoding of ordered categoricals ------------------------
    df["age_ordinal"] = df["Q4_age_group"].map(AGE_ORDER)

    # --- 3. Derived / aggregated features -----------------------------------
    pain_cols = [c for c in df.columns if c.startswith("Q13_pain_") and c != "Q13_pain_none"]
    service_cols = [c for c in df.columns if c.startswith("Q10_") and c != "Q10_none"]
    feature_cols = [c for c in df.columns if c.startswith("Q15_feat_")]
    app_feat_cols = [c for c in df.columns if c.startswith("Q24_app_") and c != "Q24_app_none_prefer_whatsapp"]

    df["num_pain_points"] = df[pain_cols].sum(axis=1)
    df["num_services_used"] = df[service_cols].sum(axis=1)
    df["num_features_wanted"] = df[feature_cols].sum(axis=1)
    df["num_app_features_wanted"] = df[app_feat_cols].sum(axis=1)
    # spend gap: how much upside exists between what they pay today and what
    # they'd be willing to pay -- a meaningful business signal, not just a raw column
    df["spend_to_wtp_gap"] = df["Q19_max_wtp_aed"] - df["Q18_current_spend_aed"]

    derived_continuous = ["num_pain_points", "num_services_used",
                           "num_features_wanted", "num_app_features_wanted",
                           "spend_to_wtp_gap"]

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
        "numeric_cols": continuous_cols + ordinal_cols,  # both get scaled
    }
    return X, y, meta


def build_preprocessor(meta):
    """Build a ColumnTransformer: scale numeric, one-hot nominal, passthrough binary."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), meta["numeric_cols"]),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), meta["nominal_cols"]),
            ("binary", "passthrough", meta["binary_cols"]),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor, meta):
    """Recover readable feature names after a ColumnTransformer has been fit."""
    numeric_names = meta["numeric_cols"]
    nominal_encoder = preprocessor.named_transformers_["nominal"]
    nominal_names = list(nominal_encoder.get_feature_names_out(meta["nominal_cols"]))
    binary_names = meta["binary_cols"]
    return numeric_names + nominal_names + binary_names
