"""
Part 2: learning curves + Fairlearn audits for Task 1 models.

Bank Marketing Part 2 EXCLUDES call duration (post-call leakage).
Learning curves plot BOTH training and validation PR-AUC.
Fairlearn reports include subgroup n and observed positive outcome rates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "part2_figures"
IMG_DIR = ROOT / "img"
OUTPUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
# Part 2 Bank Marketing: exclude post-call duration (leakage for pre-call targeting)
EXCLUDE_BANK_DURATION = True


def get_file(*names: str) -> Path:
    for name in names:
        path = ROOT / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {names}")


def load_bank() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(get_file("casestudiesDS1.csv", "casestudiesDS1(1).csv"), sep=";")
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    if EXCLUDE_BANK_DURATION and "duration" in X.columns:
        X = X.drop(columns=["duration"])
    return X, y


def load_credit() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(get_file("casestudiesDS2.csv", "casestudiesDS2(1).csv"), header=1)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    y = df["default payment next month"].astype(int)
    X = df.drop(columns=["default payment next month"])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    for c in ["SEX", "EDUCATION", "MARRIAGE"]:
        X[c] = X[c].astype("category")
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = X.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    categorical, numeric = split_columns(X)
    transformers = []
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric)
        )
    return ColumnTransformer(transformers)


def make_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
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
    return Pipeline([("pre", make_preprocessor(X, scale_numeric=False)), ("clf", model)])


def make_mlp(X: pd.DataFrame) -> Pipeline:
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
    return Pipeline([("pre", make_preprocessor(X, scale_numeric=True)), ("clf", model)])


def save_figure(filename: str) -> None:
    for directory in (OUTPUT_DIR, IMG_DIR):
        plt.savefig(directory / filename, dpi=300, bbox_inches="tight")


def create_learning_curve(X, y, dataset_name: str, output_filename: str) -> pd.DataFrame:
    print(f"\nCreating learning curve: {dataset_name}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
    models = {"XGBoost": make_xgboost(X, y), "MLP": make_mlp(X)}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {
        "XGBoost": {"train": "#1f4e79", "val": "#2f6fed"},
        "MLP": {"train": "#6b4f2a", "val": "#c47a12"},
    }
    rows = []

    for model_name, model in models.items():
        sizes, train_scores, val_scores = learning_curve(
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
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
        c = styles[model_name]

        ax.plot(sizes, train_mean, marker="s", linestyle="--", color=c["train"],
                label=f"{model_name} training")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        color=c["train"], alpha=0.12)
        ax.plot(sizes, val_mean, marker="o", linestyle="-", color=c["val"],
                label=f"{model_name} validation")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        color=c["val"], alpha=0.15)

        for i in range(len(sizes)):
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "training_samples": int(sizes[i]),
                    "train_pr_auc_mean": float(train_mean[i]),
                    "train_pr_auc_std": float(train_std[i]),
                    "validation_pr_auc_mean": float(val_mean[i]),
                    "validation_pr_auc_std": float(val_std[i]),
                    "train_val_gap": float(train_mean[i] - val_mean[i]),
                }
            )

    note = ""
    if "Bank" in dataset_name and EXCLUDE_BANK_DURATION:
        note = " (duration excluded)"
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_title(f"Learning Curve — {dataset_name}{note}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(output_filename)
    plt.close()

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / output_filename.replace(".png", ".csv"), index=False)
    print(results)
    return results


def fairness_audit(X, y, sensitive_feature: str, dataset_name: str, output_filename: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = make_xgboost(X_train, y_train)
    model.fit(X_train, y_train)
    predictions = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    if sensitive_feature == "AGE_GROUP":
        sensitive = pd.cut(
            X_test["age"],
            bins=[-np.inf, 29, 59, np.inf],
            labels=["<30", "30-59", "60+"],
        )
        sensitive.name = "Age group"
        group_col = "Age group"
    else:
        sensitive = X_test[sensitive_feature].astype(int).map({1: "Male", 2: "Female"})
        sensitive.name = "Sex"
        group_col = "Sex"

    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "recall": recall_score,
    }
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=sensitive,
    )

    audit_df = pd.DataFrame(
        {
            "group": sensitive.astype(str).to_numpy(),
            "y_true": np.asarray(y_test),
            "y_pred": predictions,
        }
    )
    group_stats = (
        audit_df.groupby("group", observed=False)
        .agg(
            n=("y_true", "size"),
            positive_outcome_rate=("y_true", "mean"),
            predicted_positive_rate=("y_pred", "mean"),
        )
        .reset_index()
    )

    by_group = metric_frame.by_group.reset_index()
    # Normalise sensitive column name
    sens_col = by_group.columns[0]
    by_group = by_group.rename(columns={sens_col: "group"})
    by_group["group"] = by_group["group"].astype(str)
    group_stats["group"] = group_stats["group"].astype(str)
    merged = group_stats.merge(by_group, on="group", how="left")

    dp_difference = demographic_parity_difference(
        y_test, predictions, sensitive_features=sensitive
    )
    eo_difference = equalized_odds_difference(
        y_test, predictions, sensitive_features=sensitive
    )

    print(f"\n===== {dataset_name} =====")
    print(merged)
    print("Demographic parity difference:", round(dp_difference, 4))
    print("Equalized odds difference:", round(eo_difference, 4))

    merged.to_csv(
        OUTPUT_DIR / output_filename.replace(".png", "_metrics.csv"), index=False
    )
    pd.DataFrame(
        {
            "dataset": [dataset_name],
            "demographic_parity_difference": [dp_difference],
            "equalized_odds_difference": [eo_difference],
            "bank_duration_excluded": [
                EXCLUDE_BANK_DURATION if "Bank" in dataset_name else None
            ],
        }
    ).to_csv(OUTPUT_DIR / output_filename.replace(".png", "_summary.csv"), index=False)

    # Combined fairness figure: rates + annotation of n / base rate
    plot_data = merged.set_index("group")[
        ["selection_rate", "true_positive_rate", "false_positive_rate"]
    ]
    plot_data.columns = ["Selection rate", "True positive rate", "False positive rate"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot_data.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(group_col)
    title = f"Fairlearn Audit — {dataset_name}"
    if "Bank" in dataset_name and EXCLUDE_BANK_DURATION:
        title += " (duration excluded)"
    ax.set_title(title)
    ax.legend(title="Metric")
    plt.xticks(rotation=0)

    # Annotate n and base rate above bars cluster
    for i, row in merged.iterrows():
        ax.text(
            i,
            1.02,
            f"n={int(row['n'])}\nbase={row['positive_outcome_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(output_filename)
    plt.close()

    # Also write a clean latex-ready table csv
    table = merged[
        [
            "group",
            "n",
            "positive_outcome_rate",
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
            "accuracy",
        ]
    ].copy()
    table.to_csv(
        OUTPUT_DIR / output_filename.replace(".png", "_table.csv"), index=False
    )
    return merged, dp_difference, eo_difference


def main():
    print("Loading datasets...")
    print(f"EXCLUDE_BANK_DURATION = {EXCLUDE_BANK_DURATION}")
    bank_X, bank_y = load_bank()
    credit_X, credit_y = load_credit()
    print("Bank features:", list(bank_X.columns))
    print("Bank:", bank_X.shape, "pos rate:", round(float(bank_y.mean()), 4))
    print("Credit:", credit_X.shape, "pos rate:", round(float(credit_y.mean()), 4))

    create_learning_curve(
        bank_X,
        bank_y,
        "Bank Marketing",
        "figure1_bank_learning_curve.png",
    )
    create_learning_curve(
        credit_X,
        credit_y,
        "Credit Card Default",
        "figure2_credit_learning_curve.png",
    )
    fairness_audit(
        bank_X,
        bank_y,
        sensitive_feature="AGE_GROUP",
        dataset_name="Bank Marketing",
        output_filename="figure3_bank_age_fairness.png",
    )
    fairness_audit(
        credit_X,
        credit_y,
        sensitive_feature="SEX",
        dataset_name="Credit Card Default",
        output_filename="figure4_credit_sex_fairness.png",
    )
    # Overleaf-friendly aliases matching user's current include names
    for src, dst in [
        ("figure1_bank_learning_curve.png", "image.png"),
        ("figure2_credit_learning_curve.png", "image2.png"),
        ("figure3_bank_age_fairness.png", "image3.png"),
        ("figure4_credit_sex_fairness.png", "image4.png"),
    ]:
        data = (OUTPUT_DIR / src).read_bytes()
        (IMG_DIR / dst).write_bytes(data)
        (ROOT / dst).write_bytes(data)
    print("\nDone. Outputs in", OUTPUT_DIR, "and", IMG_DIR)


if __name__ == "__main__":
    main()
