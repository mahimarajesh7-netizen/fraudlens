"""SHAP explainability layer on top of the trained XGBoost model.

TreeExplainer instead of a model-agnostic method (KernelExplainer, LIME): XGBoost is a tree
ensemble, so TreeExplainer computes *exact* Shapley values in polynomial time by walking the
trees directly, rather than approximating them by sampling perturbed inputs. For a fraud model
that has to justify a flagged transaction to an analyst (and potentially a regulator), an exact,
deterministic attribution is worth more than a faster but approximate one.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class FraudExplainer:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_DIR / "xgb_fraud_model.json"))
        with open(MODEL_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_proba(self, row_df: pd.DataFrame) -> float:
        dmat = xgb.DMatrix(row_df[self.feature_cols], enable_categorical=True)
        return float(self.model.predict(dmat)[0])

    def explain_row(self, row_df: pd.DataFrame, top_n: int = 6) -> dict:
        """Return the model's fraud probability for one transaction plus its top SHAP drivers."""
        proba = self.predict_proba(row_df)

        # Pass a pre-built DMatrix (not a raw DataFrame) so SHAP's internal DMatrix construction
        # is skipped entirely — it doesn't forward enable_categorical, which our category-dtype
        # columns (ProductCD, card4, M1-M9, etc.) require.
        dmat = xgb.DMatrix(row_df[self.feature_cols], enable_categorical=True)
        shap_values = self.explainer.shap_values(dmat)
        if isinstance(shap_values, list):  # older shap versions return a list for binary clf
            shap_values = shap_values[1]
        shap_row = shap_values[0] if shap_values.ndim > 1 else shap_values

        contributions = []
        for feat, sv in zip(self.feature_cols, shap_row):
            val = row_df.iloc[0][feat]
            if pd.isna(val):
                val = "missing"
            contributions.append({"feature": feat, "value": val, "shap": float(sv)})

        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)

        return {
            "fraud_probability": proba,
            "base_value": float(self.explainer.expected_value),
            "top_drivers": contributions[:top_n],
        }

    def global_importance(self, sample_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Mean |SHAP value| per feature across a sample — the global, dataset-level view that
        complements the per-transaction local explanations above."""
        dmat = xgb.DMatrix(sample_df[self.feature_cols], enable_categorical=True)
        shap_values = self.explainer.shap_values(dmat)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        out = pd.DataFrame({"feature": self.feature_cols, "mean_abs_shap": mean_abs})
        return out.sort_values("mean_abs_shap", ascending=False).head(top_n)
