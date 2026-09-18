"""Generate a global SHAP feature-importance chart and one local waterfall for a flagged fraud
transaction, saved to outputs/ for use in interview slides / talking points.
"""
import matplotlib.pyplot as plt

from src.data import load_raw, engineer_features, time_based_split
from src.explain_shap import FraudExplainer

OUT_DIR = "outputs"


def main():
    df = engineer_features(load_raw("train"))
    _, _, test_df = time_based_split(df)

    explainer = FraudExplainer()

    # Global importance
    sample = test_df.sample(n=min(3000, len(test_df)), random_state=42)
    importance = explainer.global_importance(sample, top_n=15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color="#1A7A6E")
    ax.set_title("FraudLens — Global Feature Importance (mean |SHAP|)", fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_shap_global_importance.png", dpi=150)
    print(f"Saved {OUT_DIR}/05_shap_global_importance.png")

    # One local example: highest-probability true fraud case
    fraud_rows = test_df[test_df["isFraud"] == 1]
    probs = []
    for idx in fraud_rows.head(200).index:
        row_df = test_df.loc[[idx]]
        probs.append((idx, explainer.predict_proba(row_df)))
    best_idx = max(probs, key=lambda x: x[1])[0]
    row_df = test_df.loc[[best_idx]]
    explanation = explainer.explain_row(row_df, top_n=10)

    drivers = explanation["top_drivers"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C0392B" if d["shap"] > 0 else "#1A7A6E" for d in drivers]
    ax.barh([f"{d['feature']}={d['value']}" for d in drivers][::-1],
            [d["shap"] for d in drivers][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"FraudLens — Local Explanation (TransactionID {int(test_df.loc[best_idx, 'TransactionID'])}, "
                 f"p(fraud)={explanation['fraud_probability']:.1%})", fontweight="bold", fontsize=10)
    ax.set_xlabel("SHAP contribution (+ = toward fraud)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_shap_local_example.png", dpi=150)
    print(f"Saved {OUT_DIR}/06_shap_local_example.png")


if __name__ == "__main__":
    main()
