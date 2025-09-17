import os
from pathlib import Path
import pandas as pd


def test_build_features_runs(tmp_path):
    """Run build_features on a small sampled raw dataset to keep the test fast.

    The test will copy the first N rows from each raw CSV into a temporary raw folder
    and run build_features pointing to that folder. This avoids processing the full
    production CSVs during unit tests.
    """
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    assert raw_dir.exists(), "data/raw directory not found"

    sample_dir = tmp_path / "raw_sample"
    sample_dir.mkdir()

    # small sample size
    N = 2000
    files_to_sample = ["transaction_data.csv", "customer_profiles.csv", "credit_bureau_data.csv", "credit_applications.csv", "model_predictions.csv"]
    for fname in files_to_sample:
        src = raw_dir / fname
        if not src.exists():
            continue
        dst = sample_dir / fname
        # copy header + first N lines
        with open(src, "r", encoding="utf-8", errors="ignore") as r, open(dst, "w", encoding="utf-8") as w:
            for i, line in enumerate(r):
                w.write(line)
                if i >= N:
                    break

    import sys
    sys.path.append(str(repo_root))
    from src.etl.build_features import build_features

    out = build_features(output_path=str(tmp_path / "features.parquet"), raw_dir=str(sample_dir))
    assert out is not None
    assert os.path.exists(out)

    # load and check expected columns
    if out.endswith(".parquet"):
        df = pd.read_parquet(out)
    else:
        df = pd.read_csv(out)

    assert "transaction_id" in df.columns
    assert "customer_id" in df.columns
    assert "fraud_flag" in df.columns
    # rolling features should appear
    rolling_cols = [c for c in df.columns if c.startswith("amt_")]
    assert len(rolling_cols) > 0
