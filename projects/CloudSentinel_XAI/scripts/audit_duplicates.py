from pathlib import Path
import pandas as pd

frames = []

for path in sorted(Path("data/sampled").glob("*.csv")):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df["__audit_source__"] = path.name
    frames.append(df)

df = pd.concat(frames, ignore_index=True, sort=False)
feature_cols = [
    c for c in df.columns
    if c not in {"Label", "__audit_source__"}
]

# Normalize labels only for reporting; preserve original labels.
labels = df["Label"].fillna("<MISSING>").astype(str).str.strip()
features = df[feature_cols]

print(f"Total rows: {len(df):,}")
print(f"Feature columns: {len(feature_cols)}")

# Count rows belonging to repeated feature-vector groups.
dup_mask = features.duplicated(keep=False)
print(f"Rows in repeated feature-vector groups: {int(dup_mask.sum()):,}")

# Check whether each repeated feature vector has multiple labels.
grouped = df.loc[dup_mask].groupby(
    feature_cols, dropna=False, sort=False
)["Label"].nunique(dropna=False)

conflicting = grouped[grouped > 1]
print(f"Repeated feature groups with conflicting labels: {len(conflicting):,}")

# Count duplicate feature rows that appear in more than one source file.
dup = df.loc[dup_mask, feature_cols + ["__audit_source__", "Label"]]
cross_source_groups = 0
for _, group in dup.groupby(feature_cols, dropna=False, sort=False):
    if group["__audit_source__"].nunique() > 1:
        cross_source_groups += 1

print(f"Repeated feature groups spanning multiple files: {cross_source_groups:,}")

# Report exact duplicate counts by file.
for source, group in df.groupby("__audit_source__", sort=True):
    n = int(group[feature_cols].duplicated(keep=False).sum())
    print(f"{source}: {n:,} rows in repeated feature groups")

# Write a compact summary only.
out = Path("outputs/data_audit")
out.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "feature_columns",
        "rows_in_repeated_feature_groups",
        "repeated_feature_groups_with_conflicting_labels",
        "repeated_feature_groups_spanning_multiple_files",
    ],
    "value": [
        len(df),
        len(feature_cols),
        int(dup_mask.sum()),
        len(conflicting),
        cross_source_groups,
    ],
})
summary.to_csv(out / "duplicate_audit_summary.csv", index=False)
print(f"\nSaved summary to {out / 'duplicate_audit_summary.csv'}")
