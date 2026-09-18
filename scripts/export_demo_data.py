"""Export a small, curated set of real held-out test transactions — with real model predictions
and real SHAP explanations — to JSON for the static web demo. The demo never fabricates numbers;
it replays actual pipeline output. Only the Gemini plain-language text is generated live, by the
deployed serverless function, so the demo shows a genuine API call in production rather than a
canned string.
"""
import json

from src.data import load_raw, engineer_features, time_based_split
from src.explain_shap import FraudExplainer

OUT_PATH = "web/data/demo_transactions.json"
N_FRAUD = 5
N_LEGIT = 5


def main():
    df = engineer_features(load_raw("train"))
    _, _, test_df = time_based_split(df)

    explainer = FraudExplainer()

    # Spread across probability deciles so the demo shows clear-cut and borderline cases, not
    # just the easiest examples.
    fraud_df = test_df[test_df["isFraud"] == 1].copy()
    legit_df = test_df[test_df["isFraud"] == 0].copy()

    fraud_sample = fraud_df.sample(n=min(N_FRAUD * 4, len(fraud_df)), random_state=11)
    legit_sample = legit_df.sample(n=min(N_LEGIT * 4, len(legit_df)), random_state=11)

    def score_and_pick(sample_df, n, label):
        scored = []
        for idx in sample_df.index:
            row_df = test_df.loc[[idx]]
            proba = explainer.predict_proba(row_df)
            scored.append((idx, proba))
        scored.sort(key=lambda x: x[1])
        # take an even spread across the probability range
        step = max(1, len(scored) // n)
        picked = scored[::step][:n]
        return picked

    picked = score_and_pick(fraud_sample, N_FRAUD, 1) + score_and_pick(legit_sample, N_LEGIT, 0)

    records = []
    for idx, _ in picked:
        row_df = test_df.loc[[idx]]
        explanation = explainer.explain_row(row_df, top_n=6)
        records.append({
            "transactionId": int(row_df["TransactionID"].iloc[0]),
            "actualLabel": "fraud" if row_df["isFraud"].iloc[0] == 1 else "legit",
            "fraudProbability": round(explanation["fraud_probability"], 4),
            "amount": float(row_df["TransactionAmt"].iloc[0]),
            "topDrivers": explanation["top_drivers"],
        })

    records.sort(key=lambda r: r["fraudProbability"], reverse=True)

    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {len(records)} demo transactions to {OUT_PATH}")


if __name__ == "__main__":
    main()
