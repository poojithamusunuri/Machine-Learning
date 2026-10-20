from pathlib import Path
import json
import numpy as np
import pandas as pd

source = Path("data/sampled")
out = Path("outputs/data_audit")
out.mkdir(parents=True, exist_ok=True)

files = sorted(source.glob("*.csv"))
if not files:
    raise SystemExit("No sampled CSV files found.")

frames = []
file_reports = []

for path in files:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column: {path}")

    numeric = df.select_dtypes(include=np.number)
    report = {
        "file": path.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "labels": {
            str(k): int(v)
            for k, v in df["Label"].fillna("<MISSING>").value_counts().items()
        },
        "duplicate_full_rows_within_file": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_numeric_cells": int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        ),
    }

    file_reports.append(report)
    frames.append(df)
    print(f"Audited {path.name}: {len(df):,} rows", flush=True)

combined = pd.concat(frames, ignore_index=True, sort=False)

# Exact full-row duplicates across all loaded sampled files.
cross_file_rows = combined.duplicated(keep=False)
source_counts = combined.assign(
    __audit_source__=[
        p.name for p, df in zip(files, frames) for _ in range(len(df))
    ]
)
cross_file_duplicate_rows = int(
    source_counts.duplicated(
        subset=[c for c in combined.columns if c != "__source_file__"],
        keep=False
    ).sum()
)

# Compare features separately from labels: identical features with
# different labels can be especially important to investigate.
feature_cols = [c for c in combined.columns if c != "Label"]
feature_duplicate_rows = int(combined.duplicated(
    subset=feature_cols, keep=False
).sum())

result = {
    "total_rows": int(len(combined)),
    "files": file_reports,
    "duplicate_full_rows_across_combined_sample": int(cross_file_rows.sum()),
    "duplicate_feature_rows_ignoring_label": feature_duplicate_rows,
    "label_counts": {
        str(k): int(v)
        for k, v in combined["Label"].fillna("<MISSING>").value_counts().items()
    },
}

(out / "sample_audit.json").write_text(json.dumps(result, indent=2))
print("\nAUDIT SUMMARY")
print(json.dumps(result, indent=2))
print(f"\nSaved report to {out / 'sample_audit.json'}")
