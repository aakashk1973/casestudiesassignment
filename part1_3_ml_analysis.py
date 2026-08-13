"""
Part 1.3: XGBoost vs MLP on Bank Marketing (DS1) and Credit Card Default (DS2).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_bank() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(ROOT / "casestudiesDS1.csv", sep=";")
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    # duration is known after the call — leakage for campaign targeting;i have kept for
    # exploratory insight but flag separately. Main models use full feature set
    # as provided; we also report a no-duration ablation for fair targeting insight.
    return X, y


def load_credit() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(ROOT / "casestudiesDS2.csv", header=1)
    # Drop ID
    if "ID" in raw.columns:
        raw = raw.drop(columns=["ID"])
    y = raw["default payment next month"].astype(int)
    X = raw.drop(columns=["default payment next month"])
    # PAY_2 can load as str due to CSV quirks — coerce all feature columns numeric first
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    # Treat demographic codes as categorical for one-hot
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, y


def split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    # pandas 3.x uses dtype 'str' (not 'object') for string columns
    cat = X.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    # also catch categorical-like columns set via astype("category")
    for c in X.columns:
        if c not in cat and str(X[c].dtype) in {"category", "string", "str"}:
            cat.append(c)
    num = [c for c in X.columns if c not in cat]
    return cat, num


def make_preprocessor(cat: list[str], num: list[str], scale: bool) -> ColumnTransformer:
    transformers = []
    if cat:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat,
            )
        )
    if num:
        transformers.append(
            ("num", StandardScaler() if scale else "passthrough", num)
        )
    return ColumnTransformer(transformers)


def evaluate(y_true, y_prob, y_pred) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def train_models(X: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict:
    cat, num = split_cols(X)
    pos_rate = float(y.mean())
    scale_pos = float((y == 0).sum() / max((y == 1).sum(), 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- XGBoost (tree model; scale numeric optional, OHE for cats) ---
    xgb_pre = make_preprocessor(cat, num, scale=False)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=3,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_pipe = Pipeline([("pre", xgb_pre), ("clf", xgb)])
    xgb_pipe.fit(X_train, y_train)
    xgb_prob = xgb_pipe.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    xgb_metrics = evaluate(y_test, xgb_prob, xgb_pred)

    # Feature importance on transformed names
    feature_names = xgb_pipe.named_steps["pre"].get_feature_names_out()
    importances = xgb_pipe.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    xgb_top_features = [
        {"feature": str(feature_names[i]).replace("cat__", "").replace("num__", ""),
         "importance": float(importances[i])}
        for i in top_idx
        if importances[i] > 0
    ]

    # --- MLP ---
    mlp_pre = make_preprocessor(cat, num, scale=True)
    mlp = MLPClassifier(
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
    mlp_pipe = Pipeline([("pre", mlp_pre), ("clf", mlp)])
    mlp_pipe.fit(X_train, y_train)
    mlp_prob = mlp_pipe.predict_proba(X_test)[:, 1]
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    mlp_metrics = evaluate(y_test, mlp_prob, mlp_pred)

    # Class-balanced threshold sweep for practical ops: maximize F1 on positive class
    def best_f1_threshold(y_true, y_prob):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 33):
            pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, pred, pos_label=1)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t, best_f1

    xgb_t, xgb_f1t = best_f1_threshold(y_test, xgb_prob)
    mlp_t, mlp_f1t = best_f1_threshold(y_test, mlp_prob)
    xgb_pred_t = (xgb_prob >= xgb_t).astype(int)
    mlp_pred_t = (mlp_prob >= mlp_t).astype(int)

    return {
        "dataset": dataset_name,
        "n_rows": int(len(X)),
        "n_features_raw": int(X.shape[1]),
        "positive_rate": pos_rate,
        "scale_pos_weight": scale_pos,
        "categorical_features": cat,
        "numeric_features": num,
        "xgboost": {
            **xgb_metrics,
            "top_features": xgb_top_features,
            "best_f1_threshold": xgb_t,
            "f1_at_best_threshold": float(xgb_f1t),
            "recall_at_best_threshold": float(recall_score(y_test, xgb_pred_t, pos_label=1)),
            "precision_at_best_threshold": float(
                precision_score(y_test, xgb_pred_t, pos_label=1, zero_division=0)
            ),
        },
        "mlp": {
            **mlp_metrics,
            "n_iter": int(mlp.n_iter_),
            "best_f1_threshold": mlp_t,
            "f1_at_best_threshold": float(mlp_f1t),
            "recall_at_best_threshold": float(recall_score(y_test, mlp_pred_t, pos_label=1)),
            "precision_at_best_threshold": float(
                precision_score(y_test, mlp_pred_t, pos_label=1, zero_division=0)
            ),
        },
    }


def bank_ablation_no_duration(X: pd.DataFrame, y: pd.Series) -> dict:
    """Campaign targeting without post-call duration leakage."""
    X2 = X.drop(columns=["duration"])
    out = train_models(X2, y, "Bank Marketing (no duration)")
    return {
        "xgboost_roc_auc": out["xgboost"]["roc_auc"],
        "xgboost_pr_auc": out["xgboost"]["pr_auc"],
        "xgboost_f1_positive": out["xgboost"]["f1_positive"],
        "xgboost_top_features": out["xgboost"]["top_features"][:10],
        "mlp_roc_auc": out["mlp"]["roc_auc"],
        "mlp_pr_auc": out["mlp"]["pr_auc"],
        "mlp_f1_positive": out["mlp"]["f1_positive"],
    }


def main():
    print("Loading datasets...")
    Xb, yb = load_bank()
    Xc, yc = load_credit()

    print(
        f"Bank: {Xb.shape}, positive rate={yb.mean():.3f} | "
        f"Credit: {Xc.shape}, positive rate={yc.mean():.3f}"
    )

    print("\n=== Bank Marketing (DS1) ===")
    bank = train_models(Xb, yb, "Bank Marketing (term deposit)")
    print("XGBoost ROC-AUC", bank["xgboost"]["roc_auc"], "PR-AUC", bank["xgboost"]["pr_auc"])
    print("MLP     ROC-AUC", bank["mlp"]["roc_auc"], "PR-AUC", bank["mlp"]["pr_auc"])
    print("Top XGB features:", [f["feature"] for f in bank["xgboost"]["top_features"][:8]])

    print("\n=== Bank ablation (no duration) ===")
    bank_ablation = bank_ablation_no_duration(Xb, yb)
    print(bank_ablation)

    print("\n=== Credit Card Default (DS2) ===")
    credit = train_models(Xc, yc, "Credit Card Default")
    print("XGBoost ROC-AUC", credit["xgboost"]["roc_auc"], "PR-AUC", credit["xgboost"]["pr_auc"])
    print("MLP     ROC-AUC", credit["mlp"]["roc_auc"], "PR-AUC", credit["mlp"]["pr_auc"])
    print("Top XGB features:", [f["feature"] for f in credit["xgboost"]["top_features"][:8]])

    results = {
        "case_study": "Banking customer opportunity & credit risk",
        "models": ["XGBoost", "MLP (sklearn MLPClassifier)"],
        "evaluation_metrics_justification": {
            "primary": ["pr_auc", "recall_positive", "f1_positive"],
            "secondary": ["roc_auc", "precision_positive", "balanced_accuracy"],
            "why": (
                "Both datasets are imbalanced binary classification problems in banking. "
                "Accuracy alone is misleading (majority class dominates). "
                "PR-AUC emphasises ranking quality on the rare positive class "
                "(subscribers / defaulters). Recall of the positive class aligns with "
                "business cost asymmetry: missing a subscriber wastes campaign potential; "
                "missing a defaulter creates credit loss. F1 balances precision vs recall "
                "for operational thresholding. ROC-AUC remains a secondary ranking metric."
            ),
        },
        "bank_marketing": bank,
        "bank_marketing_no_duration": bank_ablation,
        "credit_default": credit,
    }

    out_path = ROOT / "part1_3_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
