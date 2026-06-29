"""
classification_models.py
--------------------------
Trains and evaluates four classifiers on the engineered survey features:
KNN, Decision Tree, Random Forest, Gradient Boosting.

For each model we report:
  - train AND test accuracy/precision/recall/F1 (to expose overfitting --
    a model that's great on train but poor on test is unstable)
  - a full classification report (per-class precision/recall/F1)
  - confusion matrix
  - multi-class ROC curve (one-vs-rest) with AUC per class

Framework-agnostic: returns dicts/DataFrames + matplotlib figures. The
Streamlit app just calls these and displays the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import label_binarize

from .data_processing import engineer_features, build_preprocessor

RANDOM_STATE = 42

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
    """
    Train KNN / Decision Tree / Random Forest / Gradient Boosting and return
    a dict keyed by model name with the fitted pipeline + train/test data +
    predictions, ready for metric/plot generation.
    """
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

        y_train_pred = best_pipe.predict(X_train)
        y_test_pred = best_pipe.predict(X_test)

        results[name] = {
            "pipeline": best_pipe,
            "best_params": best_params,
            "y_train": y_train, "y_train_pred": y_train_pred,
            "y_test": y_test, "y_test_pred": y_test_pred,
            "classes": classes,
        }
    return results, (X_train, X_test, y_train, y_test)


def metrics_table(results):
    """Train vs. test accuracy/precision/recall/F1 (weighted) for every model -- the
    train-vs-test gap is the key 'stability' signal (a big gap = overfitting)."""
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
    """Grouped bar chart: train vs test, for each metric, across all 4 models."""
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
    """One confusion matrix heatmap per model, computed on the TEST set."""
    import seaborn as sns
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
    """
    Multi-class ROC (one-vs-rest) with AUC, one panel per model, on the TEST set.
    This is the requested 'stability' check -- a model whose ROC curve hugs the
    diagonal for any class is not reliably separating that class from the rest.
    """
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
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f"{cls} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def attach_test_probabilities(results, X_test):
    """predict_proba needs the raw X_test (pre-pipeline) -- attach it once so
    plot_roc_curves doesn't need to recompute splits."""
    for name, r in results.items():
        r["y_test_proba"] = r["pipeline"].predict_proba(X_test)
    return results


def classification_reports(results):
    """Per-class precision/recall/F1 (not just weighted averages) for depth analysis."""
    reports = {}
    for name, r in results.items():
        rep = classification_report(r["y_test"], r["y_test_pred"], output_dict=True)
        reports[name] = pd.DataFrame(rep).T.round(3)
    return reports
