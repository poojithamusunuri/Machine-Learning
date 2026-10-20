from pathlib import Path
import json
import pandas as pd

source = Path("data/sampled")
destination = Path("data/deduplicated")
audit_dir = Path("outputs/data_audit")

destination.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

reports = []

for path in sorted(source.glob("*.csv")):
    print(f"Processing {path.name}", flush=True)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column: {path}")

    feature_cols = [c for c in df.columns if c != "Label"]

    # Keep one occurrence of each feature vector within this file.
    # If duplicate feature vectors have conflicting labels, stop rather
    # than silently choosing a label.
    duplicated = df.duplicated(subset=feature_cols, keep=False)
    conflicting = (
        df.loc[duplicated]
        .groupby(feature_cols, dropna=False, sort=False)["Label"]
        .nunique(dropna=False)
    )
    if (conflicting > 1).any():
        raise ValueError(
            f"Conflicting labels among duplicate features in {path.name}"
        )

    cleaned = df.drop_duplicates(subset=feature_cols, keep="first")
    removed = len(df) - len(cleaned)

    output = destination / path.name
    cleaned.to_csv(output, index=False)

    report = {
        "source_file": path.name,
        "original_rows": int(len(df)),
        "deduplicated_rows": int(len(cleaned)),
        "removed_rows": int(removed),
        "labels_after_deduplication": {
            str(k): int(v)
            for k, v in cleaned["Label"].value_counts().items()
        },
        "output_file": str(output),
    }
    reports.append(report)

    print(
        f"  {len(df):,} -> {len(cleaned):,} rows "
        f"({removed:,} removed)",
        flush=True,
    )

summary = {
    "method": (
        "Within each source file, keep the first occurrence of each "
        "identical feature vector, excluding Label from the duplicate key."
    ),
    "files": reports,
    "total_original_rows": sum(r["original_rows"] for r in reports),
    "total_deduplicated_rows": sum(r["deduplicated_rows"] for r in reports),
    "total_removed_rows": sum(r["removed_rows"] for r in reports),
}

report_path = audit_dir / "deduplication_report.json"
report_path.write_text(json.dumps(summary, indent=2))
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print(f"\nReport saved to {report_path}")
print("Original data and Experiment 01 were not modified.")
