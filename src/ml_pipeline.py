"""
Production-ready ML pipeline for credit fraud detection using Logistic Regression, Random Forest, and XGBoost.
- Loads and preprocesses data
- Handles class imbalance
- Trains and evaluates models
- Saves best model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
from pathlib import Path

# Paths for processed features
FEATURES_PATH = Path("data") / "processed" / "features.parquet"
FALLBACK_CSV = Path("data") / "processed" / "features.csv"

# 1. Load Data
def load_data(path):
    """Load data from raw path or processed features.

    If processed features exist in `data/processed/features.parquet` they will be used.
    Otherwise, if no processed features exist the ETL builder will be invoked to create them.
    """
    # if caller supplied a path and it exists, load it
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p, parse_dates=["transaction_date"])
        except Exception:
            return pd.read_csv(p)

    # prefer processed parquet
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    if FALLBACK_CSV.exists():
        return pd.read_csv(FALLBACK_CSV, parse_dates=["transaction_date"])

    # attempt to build features via ETL
    try:
        # local import to avoid import-time overhead
        from src.etl.build_features import build_features
    except Exception:
        from etl.build_features import build_features

    out = build_features()
    if out.endswith(".parquet"):
        return pd.read_parquet(out)
    return pd.read_csv(out, parse_dates=["transaction_date"])

# 2. Preprocess Data
def preprocess(df, target_col, drop_cols=None):
    df = df.copy()
    # drop identifiers we don't want as features
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    # Basic imputation rules: numeric->median, categorical->'__missing__'
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Ensure target is numeric
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna("__missing__")

    # One-hot encode low-cardinality categoricals, leave high-cardinality as hashed features
    to_dummy = [c for c in cat_cols if df[c].nunique(dropna=False) <= 20]
    if to_dummy:
        df = pd.get_dummies(df, columns=to_dummy, drop_first=True)

    # For any remaining object columns (high-cardinality), factorize to integer codes so models/SMOTE can run.
    remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
    for col in remaining_obj:
        df[col], _ = pd.factorize(df[col])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# 3. Train/Test Split
def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def temporal_train_test_split(df: pd.DataFrame, date_col: str = "transaction_date", cutoff: str | None = None):
    """Split dataframe into train/test based on a cutoff date.

    If cutoff is None, choose the 80th percentile timestamp as cutoff.
    Returns X_train, X_test, y_train, y_test
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column {date_col} not found in dataframe for temporal split")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if cutoff is None:
        cutoff_ts = df[date_col].quantile(0.8)
    else:
        cutoff_ts = pd.to_datetime(cutoff)

    train_df = df[df[date_col] <= cutoff_ts]
    test_df = df[df[date_col] > cutoff_ts]

    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError("Temporal split produced empty train or test set; adjust cutoff or use random split")

    y_train = pd.to_numeric(train_df["fraud_flag"], errors="coerce").fillna(0).astype(int)
    y_test = pd.to_numeric(test_df["fraud_flag"], errors="coerce").fillna(0).astype(int)

    X_train = train_df.drop(columns=["fraud_flag"])
    X_test = test_df.drop(columns=["fraud_flag"])
    return X_train, X_test, y_train, y_test


def align_feature_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure X_train and X_test have identical columns. Missing columns are added with zeros.

    Returns aligned (X_train, X_test) with the same column order (train columns followed by any test-only cols).
    """
    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)
    # columns present in train but missing in test
    for c in train_cols:
        if c not in X_test.columns:
            X_test[c] = 0
    # columns present in test but missing in train
    for c in test_cols:
        if c not in X_train.columns:
            X_train[c] = 0
    # ensure same column order
    final_cols = list(X_train.columns)
    X_train = X_train[final_cols]
    X_test = X_test[final_cols]
    return X_train, X_test

# 4. Handle Imbalance
def balance_data(X, y):
    # SMOTE requires at least (k_neighbors + 1) minority samples (default k=5 -> need 6)
    try:
        value_counts = pd.Series(y).value_counts()
        if len(value_counts) < 2:
            # single class only; nothing to do
            return X, y
        minority_count = value_counts.min()
        if minority_count <= 5:
            # fallback to simple random oversampling to avoid SMOTE errors on tiny classes
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res
        else:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res
    except Exception as e:
        print(f"Warning: imbalance handling failed ({e}), returning original data")
        return X, y

# 5. Model Training/Evaluation
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    # try to add XGBoost if available; keep training robust if not
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    except Exception as e:
        print(f"XGBoost not available or failed to import: {e}. Skipping XGBoost model.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_auc = 0
    best_model = None
    results = {}

    # hyperparameter search spaces
    rf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    xgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    # determine if test has both classes
    test_has_both = len(np.unique(y_test)) > 1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        param_dist = None
        if name == 'RandomForest':
            param_dist = rf_param_dist
        # run randomized search for RF
        if param_dist is not None:
            rs = RandomizedSearchCV(model, param_dist, n_iter=4, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
            rs.fit(X_train_scaled, y_train)
            best = rs.best_estimator_
            print(f"Best params for {name}: {rs.best_params_}")
            trained = best
        else:
            model.fit(X_train_scaled, y_train)
            trained = model

        # store trained model in results
        results[name] = {'model': trained}

        # evaluate
        try:
            y_proba = trained.predict_proba(X_test_scaled)[:, 1]
            y_pred = trained.predict(X_test_scaled)
        except Exception:
            # some models may not implement predict_proba
            y_proba = trained.decision_function(X_test_scaled)
            y_pred = (y_proba > 0.5).astype(int)

        # compute AUC; if test has only one class, fall back to CV AUC on training set
        if test_has_both:
            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = float('nan')
        else:
            # compute cross-validated AUC on training set as a fallback
            try:
                cv_scores = cross_val_score(trained, X_train_scaled, y_train, scoring='roc_auc', cv=3)
                auc = float(np.mean(cv_scores))
            except Exception:
                auc = float('nan')
        print(f'AUC: {auc:.4f}')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

        # save results
        results[name]['auc'] = float(auc) if not np.isnan(auc) else None
        results[name]['classification_report'] = classification_report(y_test, y_pred, output_dict=True)

        # update best
        # select best by auc when available
        try:
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_model = (name, trained, scaler)
        except Exception:
            # auc may be None
            pass

    # attempt to train and evaluate XGBoost separately (if available)
    try:
        from xgboost import XGBClassifier
        print('\nTraining XGBoost with RandomizedSearchCV...')
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=4, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
        rs_xgb.fit(X_train_scaled, y_train)
        best_xgb = rs_xgb.best_estimator_
        y_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_xgb.predict(X_test_scaled)
        auc = roc_auc_score(y_test, y_proba)
        print(f'XGBoost AUC: {auc:.4f}')
        results['XGBoost'] = {'auc': float(auc), 'model': best_xgb, 'classification_report': classification_report(y_test, y_pred, output_dict=True)}
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_model = ('XGBoost', best_xgb, scaler)
    except Exception as e:
        print(f"XGBoost tuning skipped: {e}")

    # Save a simple results.json
    try:
        with open('models/training_results.json', 'w') as fh:
            json.dump({k: {k2: v2 for k2, v2 in v.items() if k2 != 'model'} for k, v in results.items()}, fh, indent=2)
    except Exception:
        pass

    # Try SHAP explanation for best model (only if we found one)
    if best_model is not None:
        try:
            import shap
            best_name, best_estimator, best_scaler = best_model
            print(f"Computing SHAP for best model: {best_name}")
            # use a small subset to compute SHAP values to keep runtime reasonable
            small = X_test.sample(n=min(200, len(X_test)), random_state=42)
            small_scaled = best_scaler.transform(small)
            explainer = shap.Explainer(best_estimator.predict_proba if hasattr(best_estimator, 'predict_proba') else best_estimator.predict, small_scaled)
            shap_vals = explainer(small_scaled)
            # save a small summary
            try:
                shap.summary_plot(shap_vals, small, show=False)
                # save figure to file
                import matplotlib.pyplot as plt
                plt.savefig('models/shap_summary.png')
                plt.close()
            except Exception:
                pass
        except Exception as e:
            print(f"SHAP not available or failed: {e}")
    else:
        print("No best model selected; skipping SHAP explanation")

    # if still no best model selected, pick the first trained model as fallback
    if best_model is None and len(results) > 0:
        first_name = next(iter(results))
        first_model = results[first_name].get('model')
        if first_model is not None:
            best_model = (first_name, first_model, scaler)

    return best_model

# 6. Save Model
def save_model(model_tuple, path):
    name, model, scaler = model_tuple
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f'Saved best model ({name}) to {path}')

if __name__ == "__main__":
    data_path = Path("data") / "raw" / "transaction_data.csv"
    target_col = "fraud_flag"
    drop_cols = ["transaction_id", "transaction_date"]
    df = load_data(data_path)
    X, y = preprocess(df, target_col, drop_cols)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    best_model = train_and_evaluate(X_train_bal, y_train_bal, X_test, y_test)
    save_model(best_model, "models/best_fraud_model.joblib")
