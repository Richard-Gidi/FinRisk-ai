"""Build processed features from raw CSVs.

Produces data/processed/features.parquet which the training pipeline will consume.
The implementation is conservative: it performs safe joins on customer_id, computes a
few aggregated transaction-level features and merges customer/bureau/application signals.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import numpy as np


RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)


def load_raw_tables(raw_dir: Optional[str] = None) -> dict:
    base = raw_dir or RAW_DIR
    tables = {}
    tables["transactions"] = pd.read_csv(os.path.join(base, "transaction_data.csv"), parse_dates=["transaction_date"], low_memory=False)
    tables["customers"] = pd.read_csv(os.path.join(base, "customer_profiles.csv"), parse_dates=["last_activity_date"], low_memory=False)
    tables["bureau"] = pd.read_csv(os.path.join(base, "credit_bureau_data.csv"), low_memory=False)
    tables["applications"] = pd.read_csv(os.path.join(base, "credit_applications.csv"), parse_dates=["application_date"], low_memory=False)
    # model_predictions is optional
    pred_path = os.path.join(base, "model_predictions.csv")
    if os.path.exists(pred_path):
        tables["predictions"] = pd.read_csv(pred_path, parse_dates=["prediction_date"], low_memory=False)
    else:
        tables["predictions"] = None
    return tables


def build_customer_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    # ensure correct dtypes
    df = transactions.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # basic aggregates per customer (last 30/90/365 days relative to transaction_date requires windowing; we compute global aggregates for now)
    aggs = df.groupby("customer_id").agg(
        tx_count=("transaction_id", "nunique"),
        tx_sum=("amount", "sum"),
        tx_mean=("amount", "mean"),
        tx_std=("amount", "std"),
        unique_devices=("device_info", "nunique"),
        merchant_categories=("merchant_category", lambda x: x.nunique()),
    )
    aggs = aggs.reset_index()
    # fill na for std
    aggs["tx_std"] = aggs["tx_std"].fillna(0.0)
    return aggs


def add_time_window_features(tx: pd.DataFrame, windows_days=(7, 30, 90, 365)) -> pd.DataFrame:
    """Compute rolling time-window features per transaction for each customer.

    Adds columns like `amt_sum_7d`, `amt_count_30d`, `amt_mean_90d` that summarize the
    customer's historical behaviour in the specified window up to and including the transaction.
    """
    df = tx.copy()
    # ensure datetime
    if "transaction_date" not in df.columns:
        raise ValueError("transaction_date column required for time-window features")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.sort_values(["customer_id", "transaction_date"]).set_index("transaction_date")

    # prepare a container for new features
    new_cols = {}
    grp = df.groupby("customer_id")["amount"]

    for days in windows_days:
        window = f"{days}D"
        sum_s = grp.rolling(window).sum()
        cnt_s = grp.rolling(window).count()
        mean_s = grp.rolling(window).mean()
        max_s = grp.rolling(window).max()

        # assign back to df indexed by transaction_date
        df[f"amt_sum_{days}d"] = sum_s.reset_index(level=0, drop=True)
        df[f"amt_count_{days}d"] = cnt_s.reset_index(level=0, drop=True)
        df[f"amt_mean_{days}d"] = mean_s.reset_index(level=0, drop=True)
        df[f"amt_max_{days}d"] = max_s.reset_index(level=0, drop=True)

    # fill nas with zeros for counts and sums
    for c in list(df.columns):
        if str(c).startswith("amt_"):
            df[c] = df[c].fillna(0.0)

    return df.reset_index()


def build_features(output_path: Optional[str] = None, raw_dir: Optional[str] = None) -> str:
    """Builds the feature table and writes it to disk. Returns the output path.

    The function will:
    - read raw CSVs
    - compute customer-level aggregates from transactions
    - merge customers, bureau, and last application record per customer
    - join the aggregates onto the transaction-level table so each transaction has customer features
    - write `features.parquet` to `data/processed`
    """
    tables = load_raw_tables(raw_dir)

    tx = tables["transactions"].copy()
    cust = tables["customers"].copy()
    bureau = tables["bureau"].copy()
    apps = tables["applications"].copy()
    preds = tables.get("predictions")

    # Basic cleaning
    # Normalize column names to lower
    tx.columns = [c.lower() for c in tx.columns]
    cust.columns = [c.lower() for c in cust.columns]
    bureau.columns = [c.lower() for c in bureau.columns]
    apps.columns = [c.lower() for c in apps.columns]
    if preds is not None:
        preds.columns = [c.lower() for c in preds.columns]

    # Convert amounts
    tx["amount"] = pd.to_numeric(tx.get("amount", pd.Series(dtype=float)), errors="coerce")

    # Target
    target_col = "fraud_flag"
    if target_col not in tx.columns:
        raise ValueError(f"Expected target column '{target_col}' in transactions")

    # Customer aggregates
    cust_aggs = build_customer_aggregates(tx)
    cust_aggs.columns = [c.lower() for c in cust_aggs.columns]

    # Prepare bureau keyed by customer_id: take most recent by default (if duplicates)
    if "customer_id" in bureau.columns:
        # numeric conversion
        for col in ["credit_score", "credit_history_length", "number_of_accounts", "total_credit_limit", "credit_utilization", "payment_history", "public_records"]:
            if col in bureau.columns:
                bureau[col] = pd.to_numeric(bureau[col], errors="coerce")
        # keep as is; if multiple rows per customer pick max credit_score and most recent isn't present; just aggregate
        bureau_agg = bureau.groupby("customer_id").agg(
            credit_score_bureau=("credit_score", "max"),
            credit_history_length=("credit_history_length", "max"),
            number_of_accounts=("number_of_accounts", "max"),
            total_credit_limit=("total_credit_limit", "max"),
            credit_utilization=("credit_utilization", "mean"),
            payment_history=("payment_history", "mean"),
            public_records=("public_records", "max"),
        ).reset_index()
    else:
        bureau_agg = pd.DataFrame(columns=["customer_id"])

    # Applications: keep last application per customer
    apps["application_date"] = pd.to_datetime(apps["application_date"], errors="coerce")
    apps_sorted = apps.sort_values(["customer_id", "application_date"]).drop_duplicates("customer_id", keep="last")
    apps_small = apps_sorted[["customer_id", "loan_amount", "loan_purpose", "employment_status", "annual_income", "debt_to_income_ratio", "credit_score", "application_status", "default_flag"]].copy()
    # numeric conversions
    apps_small["loan_amount"] = pd.to_numeric(apps_small["loan_amount"], errors="coerce")
    apps_small["annual_income"] = pd.to_numeric(apps_small["annual_income"], errors="coerce")
    apps_small["debt_to_income_ratio"] = pd.to_numeric(apps_small["debt_to_income_ratio"], errors="coerce")
    apps_small["credit_score_app"] = pd.to_numeric(apps_small.get("credit_score", pd.Series(dtype=float)), errors="coerce")

    # Merge features onto transactions (left join so we keep all transactions)
    feat = tx.merge(cust_aggs, on="customer_id", how="left")
    feat = feat.merge(cust.add_prefix("cust_"), left_on="customer_id", right_on="cust_customer_id", how="left")
    # drop duplicate customer_id column introduced by prefix
    if "cust_customer_id" in feat.columns:
        feat = feat.drop(columns=["cust_customer_id"])

    feat = feat.merge(bureau_agg, on="customer_id", how="left")
    feat = feat.merge(apps_small, on="customer_id", how="left")

    # optionally merge predictions (most recent)
    if preds is not None:
        preds["prediction_date"] = pd.to_datetime(preds.get("prediction_date"), errors="coerce")
        preds_sorted = preds.sort_values(["customer_id", "prediction_date"]).drop_duplicates("customer_id", keep="last")
        preds_small = preds_sorted[["customer_id", "risk_score", "fraud_probability"]].copy()
        preds_small["risk_score"] = pd.to_numeric(preds_small.get("risk_score", pd.Series(dtype=float)), errors="coerce")
        preds_small["fraud_probability"] = pd.to_numeric(preds_small.get("fraud_probability", pd.Series(dtype=float)), errors="coerce")
        feat = feat.merge(preds_small, on="customer_id", how="left")

    # Feature engineering: simple derived columns
    feat["amt_to_income"] = feat["amount"] / (feat.get("cust_annual_income", pd.Series(np.nan)).astype(float) + 1e-6)
    feat["tx_count_per_customer"] = feat.get("tx_count", 0)
    feat["avg_tx_amount"] = feat.get("tx_mean", np.nan)

    # Add rolling window aggregates per transaction
    try:
        rolling = add_time_window_features(tx)
        # merge rolling features (they are at transaction level indexed by transaction_id + date)
        rolling_cols = [c for c in rolling.columns if c.startswith("amt_")]
        rolling_subset = rolling[["transaction_id"] + rolling_cols]
        feat = feat.merge(rolling_subset, on="transaction_id", how="left")
    except Exception as e:
        print(f"Warning: could not compute rolling features: {e}")

    # Keep a reduced set of columns to keep output size reasonable
    keep_cols = [
        "transaction_id",
        "customer_id",
        "transaction_date",
        "amount",
        "merchant_category",
        "transaction_type",
        "location",
        "device_info",
        "fraud_flag",
        # aggregates & customer
        "tx_count",
        "tx_sum",
        "tx_mean",
        "tx_std",
        "unique_devices",
        "merchant_categories",
        "cust_age",
        "cust_annual_income",
        "cust_employment_status",
        "relationship_value",
        # bureau
        "credit_score_bureau",
        "credit_history_length",
        "number_of_accounts",
        "total_credit_limit",
        "credit_utilization",
        "payment_history",
        # applications
        "loan_amount",
        "loan_purpose",
        "employment_status",
        "annual_income",
        "debt_to_income_ratio",
        "credit_score_app",
        "application_status",
        "default_flag",
        # preds
        "risk_score",
        "fraud_probability",
        # derived
        "amt_to_income",
        "tx_count_per_customer",
        "avg_tx_amount",
    ]

    # Keep only the columns that exist in feat
    keep_cols = [c for c in keep_cols if c in feat.columns]
    features = feat[keep_cols].copy()

    # Convert obvious numerics
    for col in ["amount", "tx_count", "tx_sum", "tx_mean", "tx_std", "unique_devices", "merchant_categories", "cust_annual_income", "credit_score_bureau", "loan_amount", "annual_income", "debt_to_income_ratio", "risk_score", "fraud_probability", "amt_to_income"]:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Ensure transaction_date is datetime
    if "transaction_date" in features.columns:
        features["transaction_date"] = pd.to_datetime(features["transaction_date"], errors="coerce")

    # Write to parquet for compactness; fallback to csv if parquet not available
    out_path = output_path or os.path.join(PROCESSED_DIR, "features.parquet")
    try:
        features.to_parquet(out_path, index=False)
    except Exception:
        csv_out = out_path.replace(".parquet", ".csv")
        features.to_csv(csv_out, index=False)
        out_path = csv_out

    return out_path


if __name__ == "__main__":
    out = build_features()
    print(f"Wrote features to: {out}")
