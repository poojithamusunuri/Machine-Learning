from pathlib import Path
import pandas as pd

SOURCE = Path("data/raw")
DEST = Path("data/sampled")
MAX_PER_LABEL_PER_FILE = 10_000
SEED = 42

DEST.mkdir(parents=True, exist_ok=True)

for path in sorted(SOURCE.glob("*.csv")):
    print(f"\nReading labels from: {path.name}", flush=True)

    # Pass 1: count labels without loading the entire CSV into memory.
    counts = {}
    for chunk in pd.read_csv(path, usecols=["Label"], chunksize=100_000):
        labels = chunk["Label"].fillna("<MISSING>").astype(str).str.strip()
        for label, count in labels.value_counts().items():
            counts[label] = counts.get(label, 0) + int(count)

    print("Original label counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count:,}")

    # Pass 2: keep a reproducible random sample per label.
    # Reservoir sampling avoids selecting only the first rows of each class.
    import random
    rng = random.Random(SEED)
    reservoirs = {label: [] for label in counts}
    seen = {label: 0 for label in counts}

    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if "Label" not in chunk.columns:
            raise ValueError(f"No Label column in {path}")

        labels = chunk["Label"].fillna("<MISSING>").astype(str).str.strip()
        chunk = chunk.copy()
        chunk["Label"] = labels

        for label, group in chunk.groupby("Label", sort=False):
            reservoir = reservoirs[label]
            limit = min(MAX_PER_LABEL_PER_FILE, counts[label])

            for row in group.itertuples(index=False, name=None):
                seen[label] += 1
                item = row
                if len(reservoir) < limit:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen[label])
                    if j < limit:
                        reservoir[j] = item

    # Keep the original feature columns and write only the sample.
    columns = list(pd.read_csv(path, nrows=0).columns)
    sampled_frames = [
        pd.DataFrame(reservoirs[label], columns=columns)
        for label in reservoirs
        if reservoirs[label]
    ]
    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    output = DEST / path.name
    sampled.to_csv(output, index=False)

    print(f"Saved {len(sampled):,} sampled rows to {output}", flush=True)

print("\nSampling complete. Original files were not modified.")
