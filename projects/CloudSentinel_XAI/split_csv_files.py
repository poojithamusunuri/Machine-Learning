from pathlib import Path
import pandas as pd

# Change this folder if your original CSVs are somewhere else.
SOURCE_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/split_parts")

# 20 MiB target: safely below 25 MB.
MAX_BYTES = 20 * 1024 * 1024
CHUNK_ROWS = 10000

FILES = [
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_frame(frame, prefix, part_number, output_files):
    """Write a CSV part, splitting its rows further if it is too large."""
    destination = OUTPUT_DIR / f"{prefix}_part{part_number:03d}.csv"
    frame.to_csv(destination, index=False)

    size = destination.stat().st_size

    if size < MAX_BYTES:
        output_files.append(destination)
        return part_number + 1

    destination.unlink()

    if len(frame) <= 1:
        raise RuntimeError(
            f"A single row exceeds the size limit: {prefix}"
        )

    midpoint = len(frame) // 2

    part_number = split_frame(
        frame.iloc[:midpoint], prefix, part_number, output_files
    )
    part_number = split_frame(
        frame.iloc[midpoint:], prefix, part_number, output_files
    )

    return part_number


for filename in FILES:
    source = SOURCE_DIR / filename

    if not source.exists():
        raise FileNotFoundError(
            f"\nFile not found: {source}\n"
            "Put the original CSVs in data/raw/ or update SOURCE_DIR."
        )

    prefix = source.stem
    print(f"\nProcessing: {filename}")
    print(f"Original size: {source.stat().st_size / 1024**2:.2f} MiB")

    generated_parts = []
    next_part = 1
    total_rows = 0

    for chunk in pd.read_csv(
        source,
        chunksize=CHUNK_ROWS,
        low_memory=False
    ):
        total_rows += len(chunk)
        next_part = split_frame(
            chunk, prefix, next_part, generated_parts
        )

    largest = max(p.stat().st_size for p in generated_parts)

    # Verify the output size limit.
    assert all(
        p.stat().st_size < MAX_BYTES for p in generated_parts
    )

    print(f"Data rows: {total_rows:,}")
    print(f"Parts created: {len(generated_parts)}")
    print(f"Largest part: {largest / 1024**2:.2f} MiB")
    print("Size validation: PASSED")

print(f"\nAll parts saved in: {OUTPUT_DIR.resolve()}")
print("Original CSV files were not modified.")
