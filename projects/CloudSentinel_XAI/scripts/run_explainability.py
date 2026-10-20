

import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.loader import read_flow_csvs
from src.data.preprocessing import clean_frame
from src.explainability import (
    local_feature_sensitivity,
    global_permutation_sensitivity,
)

SEED = 42
MODEL_PATH = Path(
    "outputs/experiment_02/artifacts/cloudsentinel_pipeline.joblib"
)
DATA_DIR = Path("data/deduplicated")
OUT_DIR = Path("outputs/experiment_02/explanations")
CONFIG_PATH = Path("configs/experiment_02.yaml")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the already-trained Experiment 02 model.
    bundle = joblib.load(MODEL_PATH)
    prep = bundle["preprocessor"]
    detector = bundle["detector"]

    feature_names = prep.get_feature_names_out().tolist()
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Encoded feature count: {len(feature_names)}")

    # 2. Load the deduplicated sample, not the full raw CSVs.
    raw, provenance = read_flow_csvs(str(DATA_DIR))
    X, labels = clean_frame(raw)

    # Match the input columns and order used when fitting the preprocessor.
    expected_columns = list(prep.feature_names_in_)
    missing = sorted(set(expected_columns) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected_columns))

    if missing:
        raise ValueError(f"Missing model input columns: {missing}")

    if extra:
        print(f"Extra input columns ignored: {extra}")

    X = X.loc[:, expected_columns].reset_index(drop=True)
    labels = labels.astype(str).str.strip().reset_index(drop=True)

    # 3. Reconstruct the experiment's known-family split.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    family_map = yaml.safe_load(
        Path("configs/attack_family_map.yaml").read_text()
    )["family_by_label"]

    normalized_map = {
        str(label).strip().upper(): str(family).strip().upper()
        for label, family in family_map.items()
    }
    families = labels.str.upper().map(normalized_map)

    if families.isna().any():
        unknown = labels[families.isna()].unique().tolist()
        raise ValueError(f"Labels missing from family map: {unknown}")

    benign_label = str(config.get("benign_label", "BENIGN")).upper()
    benign = labels.str.upper().eq(benign_label)
    heldout = families.eq(
        str(config["heldout_family"]).strip().upper()
    )

    # Exclude the held-out family from the development validation set.
    known_mask = ~heldout
    X_known = X.loc[known_mask].reset_index(drop=True)
    y_known = (~benign.loc[known_mask]).astype(int).reset_index(drop=True)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_known,
        y_known,
        test_size=float(config["test_size"]),
        random_state=SEED,
        stratify=y_known,
    )

    validation_fraction = (
        float(config["validation_size"])
        / (1.0 - float(config["test_size"]))
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=validation_fraction,
        random_state=SEED,
        stratify=y_trainval,
    )

    print(f"Known-family validation rows: {len(X_val)}")
    print(f"Validation benign rows: {(y_val == 0).sum()}")
    print(f"Validation attack rows: {(y_val == 1).sum()}")

    # 4. Explain a bounded, reproducible validation subset.
    # This is not the held-out DOS evaluation set.
    n_local = min(300, len(X_val))
    X_local = X_val.sample(n=n_local, random_state=SEED)
    y_local = y_val.loc[X_local.index]

    X_local_encoded = prep.transform(X_local)
    if hasattr(X_local_encoded, "toarray"):
        X_local_encoded = X_local_encoded.toarray()
    X_local_encoded = np.asarray(X_local_encoded, dtype=float)

    # Use a bounded background set for local sensitivity.
    background = X_local_encoded[: min(100, len(X_local_encoded))]
    attack_positions = np.flatnonzero(y_local.to_numpy() == 1)

    if len(attack_positions) == 0:
        raise ValueError("No attack rows found in the local validation subset.")

    example_position = int(attack_positions[0])
    example_row = X_local_encoded[example_position]

    local_result = local_feature_sensitivity(
        detector.supervised,
        example_row,
        background,
        feature_names,
        top_k=12,
    )
    local_path = OUT_DIR / "local_feature_sensitivity.csv"
    local_result.to_csv(local_path, index=False)

    local_metadata = {
        "model_path": str(MODEL_PATH),
        "data_dir": str(DATA_DIR),
        "example_type": "known-family validation attack",
        "validation_subset_rows": len(X_local),
        "background_rows": len(background),
        "encoded_feature_count": len(feature_names),
        "interpretation": (
            "One-feature median-replacement sensitivity; "
            "not SHAP and not a causal explanation."
        ),
    }
    (OUT_DIR / "local_explanation_metadata.json").write_text(
        pd.Series(local_metadata).to_json(indent=2)
    )

    print(f"\nLocal sensitivity saved: {local_path}")
    print(local_result.to_string(index=False))

    # 5. Global permutation sensitivity on validation data only.
    # Restrict to at most 1,000 rows to keep the run manageable.
    n_global = min(1000, len(X_val))
    X_global = X_val.sample(n=n_global, random_state=SEED)
    y_global = y_val.loc[X_global.index]

    X_global_encoded = prep.transform(X_global)
    if hasattr(X_global_encoded, "toarray"):
        X_global_encoded = X_global_encoded.toarray()
    X_global_encoded = np.asarray(X_global_encoded, dtype=float)

    global_result = global_permutation_sensitivity(
        detector.supervised,
        X_global_encoded,
        y_global.to_numpy(),
        feature_names,
        n_repeats=3,
        seed=SEED,
        max_features=25,
    )

    global_path = OUT_DIR / "global_permutation_sensitivity.csv"
    global_result.to_csv(global_path, index=False)

    print(f"\nGlobal sensitivity saved: {global_path}")
    print(global_result.head(15).to_string(index=False))

    print("\nExplainability run completed.")
    print("No model retraining or threshold changes were performed.")
    print("The held-out DOS family was excluded from this analysis.")


if __name__ == "__main__":
    main()
