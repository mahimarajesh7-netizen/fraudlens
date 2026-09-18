"""Train and evaluate the XGBoost fraud classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data import load_raw, engineer_features, time_based_split, get_feature_cols

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_dmatrix(df, feature_cols):
    return xgb.DMatrix(df[feature_cols], label=df["isFraud"], enable_categorical=True)


def train(max_rows: int | None = None) -> dict:
    print("Loading and merging raw data...")
    df = load_raw("train")
    if max_rows:
        df = df.sample(n=max_rows, random_state=42)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    df = engineer_features(df)
    train_df, val_df, test_df = time_based_split(df)
    print(f"Split -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

    feature_cols = get_feature_cols(df)

    dtrain = build_dmatrix(train_df, feature_cols)
    dval = build_dmatrix(val_df, feature_cols)
    dtest = build_dmatrix(test_df, feature_cols)

    # scale_pos_weight reweights the loss for the minority (fraud) class directly, rather than
    # resampling rows (e.g. SMOTE). With ~430 largely-anonymized/categorical features, synthetic
    # interpolation between fraud examples risks manufacturing unrealistic points; reweighting
    # the objective keeps every training row real.
    neg, pos = (train_df["isFraud"] == 0).sum(), (train_df["isFraud"] == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance -> neg: {neg:,} pos: {pos:,} scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
    }

    evals_result = {}
    print("Training XGBoost (early stopping on validation AUC)...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=150,
        evals_result=evals_result,
        verbose_eval=200,
    )

    test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    test_auc = roc_auc_score(test_df["isFraud"], test_pred)
    test_pr_auc = average_precision_score(test_df["isFraud"], test_pred)

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Held-out TEST ROC-AUC:  {test_auc:.4f}")
    print(f"Held-out TEST PR-AUC:   {test_pr_auc:.4f}  (baseline = fraud rate = {test_df['isFraud'].mean():.4f})")

    model.save_model(str(MODEL_DIR / "xgb_fraud_model.json"))
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    metrics = {
        "test_roc_auc": test_auc,
        "test_pr_auc": test_pr_auc,
        "best_iteration": model.best_iteration,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "fraud_rate_test": float(test_df["isFraud"].mean()),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train()
