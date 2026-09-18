"""End-to-end demo: pick transactions from the held-out set, run the model, explain via SHAP,
then via Gemini. Usage: python -m scripts.run_explain [--n 3] [--fraud-only]
"""
import argparse
import os

from dotenv import load_dotenv

from src.data import load_raw, engineer_features, time_based_split
from src.explain_shap import FraudExplainer
from src.explain_gemini import explain_in_plain_language

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--fraud-only", action="store_true")
    args = parser.parse_args()

    print("Loading data and rebuilding the held-out test split...")
    df = engineer_features(load_raw("train"))
    _, _, test_df = time_based_split(df)

    if args.fraud_only:
        test_df = test_df[test_df["isFraud"] == 1]

    sample = test_df.sample(n=min(args.n, len(test_df)), random_state=7)

    explainer = FraudExplainer()

    for idx in sample.index:
        row_df = test_df.loc[[idx]]  # double brackets: keep as DataFrame, preserving dtypes
        explanation = explainer.explain_row(row_df)

        print("\n" + "=" * 70)
        print(f"TransactionID: {int(row_df['TransactionID'].iloc[0])}   Actual label: "
              f"{'FRAUD' if row_df['isFraud'].iloc[0] == 1 else 'legit'}")
        print(f"Model fraud probability: {explanation['fraud_probability']:.1%}")
        print("\nTop SHAP drivers:")
        for d in explanation["top_drivers"]:
            direction = "-> FRAUD" if d["shap"] > 0 else "-> legit"
            print(f"  {d['feature']:<25} = {str(d['value']):<20} shap={d['shap']:+.4f} {direction}")

        try:
            text = explain_in_plain_language(explanation)
            print("\nGemini plain-language explanation:")
            print(f"  {text}")
        except Exception as e:
            print(f"\n[Gemini call failed: {e}]")


if __name__ == "__main__":
    main()
