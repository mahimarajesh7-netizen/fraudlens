# FraudLens

End-to-end fraud detection with AI-generated explainability, built on the IEEE-CIS fraud dataset
(590K transactions, 434 features after merging transaction + identity tables).

## Pipeline

1. **Data** (`src/data.py`) — loads and left-joins `train_transaction.csv` + `train_identity.csv`
   on `TransactionID`, engineers time/amount/frequency features, and splits by `TransactionDT`
   (not randomly) into train/val/test to simulate predicting on genuinely future transactions.
2. **Model** (`src/train.py`) — XGBoost (`binary:logistic`), missing values left as native NaNs
   (XGBoost splits on missingness directly rather than needing imputation), class imbalance
   handled via `scale_pos_weight` rather than resampling, early stopping on a validation ROC-AUC.
3. **Explainability** (`src/explain_shap.py`) — `shap.TreeExplainer` computes exact per-transaction
   feature attributions (not an approximation), both locally (why *this* transaction was flagged)
   and globally (which features matter most across the model).
4. **Plain-language layer** (`src/explain_gemini.py`) — takes the SHAP attribution for one
   transaction and asks Gemini to translate it into a short analyst-readable explanation, grounded
   strictly in the SHAP output so it can't invent reasons the model didn't actually use.

## Results

Measured on a genuinely held-out, time-based test split (last 15% of transactions by
`TransactionDT` — never seen in training or validation):

| Metric | Value |
|---|---|
| Test ROC-AUC | **0.894** |
| Test PR-AUC | 0.534 (vs. 0.035 baseline fraud rate) |
| Validation ROC-AUC | 0.930 |
| Train ROC-AUC | 1.000 (fully memorized — expected overfit gap with 434 features) |

Top global SHAP drivers: `card1` (by far the largest), `addr1`, `card2`, `C13`, `card1_freq`,
`TransactionAmt`, `C1`, `P_emaildomain`. `card1` dominating so heavily is worth noting on its own:
it's a near-unique card/account identifier, so a meaningful share of what the model has learned is
closer to "this specific card was fraudulent in training" than a generalizable behavioral pattern —
a real limitation to be upfront about, not just a scoreboard number.

Re-run with `python -m src.train`; see `models/metrics.json` for the latest run.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place train_transaction.csv, train_identity.csv (and test_*) under data/
python -m src.train              # trains XGBoost, writes models/xgb_fraud_model.json + metrics.json
python -m scripts.run_explain --n 3   # SHAP + Gemini explanation demo on sample transactions
```

Requires a `GEMINI_API_KEY` in `.env` for the explanation layer.
