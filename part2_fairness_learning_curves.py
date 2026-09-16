from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "part2_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# DATA LOADING
# =========================================================

def get_file(*names):
    """Use first matching filename."""
    for name in names:
        path = ROOT / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find any of these files: {names}"
    )


def load_bank():
    path = get_file(
        "casestudiesDS1.csv",
        "casestudiesDS1(1).csv",
    )

    df = pd.read_csv(path, sep=";")

    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])

    return X, y


def load_credit():
    path = get_file(
        "casestudiesDS2.csv",
        "casestudiesDS2(1).csv",
    )

    df = pd.read_csv(path, header=1)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    y = df["default payment next month"].astype(int)

    X = df.drop(
        columns=["default payment next month"]
    )

    # Same processing as Task 1
    for c in X.columns:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    for c in ["SEX", "EDUCATION", "MARRIAGE"]:
        X[c] = X[c].astype("category")

    return X, y


# =========================================================
# PREPROCESSING
# =========================================================

def split_columns(X):
    categorical = X.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns.tolist()

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    return categorical, numeric


def make_preprocessor(X, scale_numeric=False):

    categorical, numeric = split_columns(X)

    transformers = []

    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical,
            )
        )

    if numeric:
        transformers.append(
            (
                "num",
                StandardScaler()
                if scale_numeric
                else "passthrough",
                numeric,
            )
        )

    return ColumnTransformer(transformers)


# =========================================================
# MODELS
# These match the Task 1 configuration
# =========================================================

def make_xgboost(X, y):

    negative = (y == 0).sum()
    positive = (y == 1).sum()

    scale_pos_weight = (
        negative / max(positive, 1)
    )

    preprocessor = make_preprocessor(
        X,
        scale_numeric=False
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("pre", preprocessor),
            ("clf", model),
        ]
    )


def make_mlp(X):

    preprocessor = make_preprocessor(
        X,
        scale_numeric=True
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("pre", preprocessor),
            ("clf", model),
        ]
    )


# =========================================================
# LEARNING CURVES
# =========================================================

def create_learning_curve(
    X,
    y,
    dataset_name,
    output_filename,
):

    print(
        f"\nCreating learning curve: "
        f"{dataset_name}"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_sizes = np.array(
        [0.10, 0.25, 0.50, 0.75, 1.00]
    )

    models = {
        "XGBoost": make_xgboost(X, y),
        "MLP": make_mlp(X),
    }

    plt.figure(figsize=(8, 5))

    rows = []

    for model_name, model in models.items():

        sizes, train_scores, val_scores = (
            learning_curve(
                model,
                X,
                y,
                train_sizes=train_sizes,
                cv=cv,
                scoring="average_precision",
                n_jobs=1,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)

        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.plot(
            sizes,
            val_mean,
            marker="o",
            label=f"{model_name} validation"
        )

        plt.fill_between(
            sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15
        )

        for i in range(len(sizes)):
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "training_samples": int(sizes[i]),
                "train_pr_auc_mean":
                    train_mean[i],
                "train_pr_auc_std":
                    train_std[i],
                "validation_pr_auc_mean":
                    val_mean[i],
                "validation_pr_auc_std":
                    val_std[i],
            })

    plt.xlabel("Number of training samples")
    plt.ylabel("PR-AUC (Average Precision)")
    plt.title(
        f"Learning Curve — {dataset_name}"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    results = pd.DataFrame(rows)

    csv_name = (
        output_filename
        .replace(".png", ".csv")
    )

    results.to_csv(
        OUTPUT_DIR / csv_name,
        index=False
    )

    print(results)

    return results


# =========================================================
# FAIRLEARN ANALYSIS
# =========================================================

def fairness_audit(
    X,
    y,
    sensitive_feature,
    dataset_name,
    output_filename,
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Keep exact Task 1 XGBoost configuration
    model = make_xgboost(
        X_train,
        y_train
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------
    # Sensitive feature
    # ---------------------------------------------

    if sensitive_feature == "AGE_GROUP":

        sensitive = pd.cut(
            X_test["age"],
            bins=[
                -np.inf,
                29,
                59,
                np.inf
            ],
            labels=[
                "<30",
                "30-59",
                "60+"
            ],
        )

        sensitive.name = "Age group"

    else:

        sensitive = X_test[
            sensitive_feature
        ].copy()

        # Convert coded values to readable names
        if sensitive_feature == "SEX":

            sensitive = sensitive.astype(
                "int"
            ).map({
                1: "Male",
                2: "Female"
            })

            sensitive.name = "Sex"

    # ---------------------------------------------
    # Fairlearn MetricFrame
    # ---------------------------------------------

    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate":
            true_positive_rate,
        "false_positive_rate":
            false_positive_rate,
        "recall": recall_score,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=sensitive,
    )

    print(
        f"\n===== {dataset_name} ====="
    )

    print("\nOverall metrics:")
    print(metric_frame.overall)

    print("\nMetrics by group:")
    print(metric_frame.by_group)

    # ---------------------------------------------
    # Fairness gaps
    # ---------------------------------------------

    dp_difference = (
        demographic_parity_difference(
            y_test,
            predictions,
            sensitive_features=sensitive,
        )
    )

    eo_difference = (
        equalized_odds_difference(
            y_test,
            predictions,
            sensitive_features=sensitive,
        )
    )

    print(
        "\nDemographic parity difference:",
        round(dp_difference, 4)
    )

    print(
        "Equalized odds difference:",
        round(eo_difference, 4)
    )

    # ---------------------------------------------
    # Dataset outcome rate by group
    # ---------------------------------------------

    audit_df = pd.DataFrame({
        "group": sensitive.astype(str),
        "y_true": np.array(y_test),
        "y_pred": predictions,
    })

    outcome_rates = (
        audit_df
        .groupby("group", observed=False)
        ["y_true"]
        .mean()
    )

    print(
        "\nObserved positive outcome rate "
        "in dataset:"
    )

    print(outcome_rates)

    # ---------------------------------------------
    # Save Fairlearn results
    # ---------------------------------------------

    group_results = (
        metric_frame.by_group.reset_index()
    )

    group_results.to_csv(
        OUTPUT_DIR /
        output_filename.replace(
            ".png",
            "_metrics.csv"
        ),
        index=False,
    )

    summary = pd.DataFrame({
        "dataset": [dataset_name],
        "demographic_parity_difference":
            [dp_difference],
        "equalized_odds_difference":
            [eo_difference],
    })

    summary.to_csv(
        OUTPUT_DIR /
        output_filename.replace(
            ".png",
            "_summary.csv"
        ),
        index=False,
    )

    # ---------------------------------------------
    # Plot
    # ---------------------------------------------

    plot_data = metric_frame.by_group[
        [
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
        ]
    ]

    plot_data.columns = [
        "Selection rate",
        "True positive rate",
        "False positive rate",
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_xlabel(
        sensitive.name
        if sensitive.name
        else "Sensitive group"
    )

    ax.set_title(
        f"Fairness Audit — {dataset_name}"
    )

    ax.legend(
        title="Metric"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return (
        metric_frame,
        dp_difference,
        eo_difference
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading datasets...")

    bank_X, bank_y = load_bank()
    credit_X, credit_y = load_credit()

    print(
        "\nBank Marketing:",
        bank_X.shape,
        "positive rate:",
        round(bank_y.mean(), 4)
    )

    print(
        "Credit Default:",
        credit_X.shape,
        "positive rate:",
        round(credit_y.mean(), 4)
    )

    # =====================================================
    # FIGURE 1
    # =====================================================

    create_learning_curve(
        bank_X,
        bank_y,
        "Bank Marketing",
        "figure1_bank_learning_curve.png",
    )

    # =====================================================
    # FIGURE 2
    # =====================================================

    create_learning_curve(
        credit_X,
        credit_y,
        "Credit Card Default",
        "figure2_credit_learning_curve.png",
    )

    # =====================================================
    # FIGURE 3
    # Fairlearn Bank Marketing — AGE
    # =====================================================

    fairness_audit(
        bank_X,
        bank_y,
        sensitive_feature="AGE_GROUP",
        dataset_name="Bank Marketing",
        output_filename=
        "figure3_bank_age_fairness.png",
    )

    # =====================================================
    # FIGURE 4
    # Fairlearn Credit Default — SEX
    # =====================================================

    fairness_audit(
        credit_X,
        credit_y,
        sensitive_feature="SEX",
        dataset_name="Credit Card Default",
        output_filename=
        "figure4_credit_sex_fairness.png",
    )

    print(
        "\nDone. Files saved in:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()