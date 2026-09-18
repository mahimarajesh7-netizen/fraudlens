"""Load, merge, and feature-engineer the IEEE-CIS fraud dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORICAL_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
] + [f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)]


def load_raw(split: str = "train") -> pd.DataFrame:
    """Load and left-join the transaction and identity tables for a split ('train' or 'test')."""
    trans = pd.read_csv(DATA_DIR / f"{split}_transaction.csv")
    ident = pd.read_csv(DATA_DIR / f"{split}_identity.csv")
    return trans.merge(ident, on="TransactionID", how="left")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time, amount, and frequency features. Leaves NaNs in place — XGBoost splits on
    missingness natively, and for this dataset missingness is itself informative (MNAR), so
    imputing would throw away signal rather than clean it up.
    """
    df = df.copy()

    # Time features (TransactionDT is seconds from an arbitrary reference point, not a real
    # timestamp — but hour-of-day and day-of-week cycles still line up with real behavior).
    df["hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
    df["day_of_week"] = (df["TransactionDT"] / (3600 * 24)).astype(int) % 7

    # Amount features: the fractional cents of TransactionAmt correlate with currency/processor
    # (round-dollar USD vs. converted foreign amounts), and log-amount tames the heavy right tail.
    df["TransactionAmt_decimal"] = ((df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 1000).round().astype(int)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Frequency encoding for high-cardinality card/address fields: how many times this exact
    # value appears in the data acts as a cheap proxy for "is this a real recurring account or
    # a one-off value," without exploding dimensionality the way one-hot encoding would.
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        if col in df.columns:
            freq = df[col].map(df[col].value_counts(dropna=False))
            df[f"{col}_freq"] = freq

    # Cast known categoricals to pandas 'category' dtype so XGBoost's native categorical
    # split-finding can use them directly, instead of label-encoding by hand.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def time_based_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Split by TransactionDT rather than randomly. A random split lets the model see rows from
    the same time window (and often the same recurring card/device) in both train and test,
    which inflates offline AUC relative to how the model performs on genuinely future
    transactions in production — a time-based split is the honest simulation of deployment.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def get_feature_cols(df: pd.DataFrame) -> list:
    drop_cols = {"TransactionID", "isFraud", "TransactionDT"}
    return [c for c in df.columns if c not in drop_cols]
