"""Full training runner.

Usage:
  python scripts/run_train.py [--temporal] [--cutoff YYYY-MM-DD] [--sample N]

By default this runs on the full processed features; use --sample to limit rows for quick runs.
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.ml_pipeline import load_data, preprocess, get_train_test, temporal_train_test_split, balance_data, train_and_evaluate, save_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal", action="store_true", help="Use temporal split")
    p.add_argument("--cutoff", type=str, default=None, help="Cutoff date for temporal split (YYYY-MM-DD)")
    p.add_argument("--sample", type=int, default=0, help="Sample N rows for quick runs")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(repo_root / 'data' / 'raw' / 'transaction_data.csv')
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42)

    if args.temporal:
        X_train, X_test, y_train, y_test = temporal_train_test_split(df, date_col='transaction_date', cutoff=args.cutoff)
        # if test set has no positive labels, fallback to stratified random split
        if y_test.sum() == 0:
            print("Temporal split produced zero positive samples in test set; falling back to stratified random split.")
            X_train, X_test, y_train, y_test = get_train_test(df.drop(columns=['fraud_flag']), df['fraud_flag'], test_size=0.2, random_state=42)
    else:
        X, y = preprocess(df, 'fraud_flag', drop_cols=['transaction_id', 'transaction_date'])
        X_train, X_test, y_train, y_test = get_train_test(X, y)

    # If we used temporal split, ensure preprocessing runs consistently
    if args.temporal:
        # recombine X_train/X_test with fraud_flag for consistent preprocess
        train_df = X_train.copy()
        train_df['fraud_flag'] = y_train
        test_df = X_test.copy()
        test_df['fraud_flag'] = y_test
        # preprocess both
        X_train_p, y_train_p = preprocess(train_df, 'fraud_flag', drop_cols=['transaction_id', 'transaction_date'] if 'transaction_date' in train_df.columns else None)
        X_test_p, y_test_p = preprocess(test_df, 'fraud_flag', drop_cols=['transaction_id', 'transaction_date'] if 'transaction_date' in test_df.columns else None)
        # align features
        from src.ml_pipeline import align_feature_columns
        X_train_al, X_test_al = align_feature_columns(X_train_p, X_test_p)
        X_train, X_test, y_train, y_test = X_train_al, X_test_al, y_train_p, y_test_p

    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    best = train_and_evaluate(X_train_bal, y_train_bal, X_test, y_test)
    # save the best model if available
    if best is None:
        print("Training completed but no best model was selected. No model will be saved.")
    else:
        save_model(best, repo_root / 'models' / 'best_fraud_model.joblib')


if __name__ == '__main__':
    main()
