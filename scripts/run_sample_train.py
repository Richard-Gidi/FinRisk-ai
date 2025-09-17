"""Run a sampled end-to-end training to limit runtime for CI/dev.

Usage: python scripts/run_sample_train.py [sample_size]
"""
import os
import sys
from pathlib import Path

sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('SAMPLE_SIZE', 5000))

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.ml_pipeline import load_data, preprocess, get_train_test, balance_data, train_and_evaluate, save_model

df = load_data(repo_root / 'data' / 'raw' / 'transaction_data.csv')
if sample_size and len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print('Running train on sample size:', len(df))
X, y = preprocess(df, 'fraud_flag', drop_cols=['transaction_id', 'transaction_date'])
X_train, X_test, y_train, y_test = get_train_test(X, y, test_size=0.2)
X_train_bal, y_train_bal = balance_data(X_train, y_train)
best = train_and_evaluate(X_train_bal, y_train_bal, X_test, y_test)
print('Best model:', best[0])
save_model(best, repo_root / 'models' / 'best_fraud_model_sample.joblib')
