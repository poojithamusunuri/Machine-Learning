
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.data.loader import read_flow_csvs
from src.data.preprocessing import clean_frame


SEED = 42
CONFIG_PATH = Path("configs/experiment_02.yaml")
MODEL_PATH = Path(
    "outputs/experiment_02/artifacts/cloudsentinel_pipeline.joblib"
)
DATA_DIR = Path("data/deduplicated")
OUT_DIR = Path("outputs/experiment_02/error_analysis")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the existing model and configuration.
    bundle = joblib.load(MODEL_PATH)
    prep = bundle["preprocessor"]
    detector = bundle["detector"]
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    family_map = yaml.safe_load(
        Path("configs/attack_family_map.yaml").read_text()
    )["family_by_label"]

    # Load and clean the deduplicated dataset.
    raw, _ = read_flow_csvs(str(DATA_DIR))
    X, labels = clean_frame(raw)

    labels = labels.astype(str).str.strip().reset_index(drop=True)
    X = X.loc[:, list(prep.feature_names_in_)].reset_index(drop=True)

    mapping = {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in family_map.items()
    }
    families = labels.str.upper().map(mapping)

    if families.isna().any():
        unknown = labels[families.isna()].unique().tolist()
        raise ValueError(f"Labels missing from family map: {unknown}")

    benign = labels.str.upper().eq(
        str(cfg.get("benign_label", "BENIGN")).upper()
    )
    heldout = families.eq(
        str(cfg["heldout_family"]).strip().upper()
    )

    # Exclude the held-out family from known-family error analysis.
    known = ~heldout
    X_known = X.loc[known].reset_index(drop=True)
    y_known = (~benign.loc[known]).astype(int).reset_index(drop=True)
    labels_known = labels.loc[known].reset_index(drop=True)
    families_known = families.loc[known].reset_index(drop=True)

    # Reproduce the deterministic Experiment 02 test split.
    positions = np.arange(len(X_known))
    _, test_positions = train_test_split(
        positions,
        test_size=float(cfg["test_size"]),
        random_state=SEED,
        stratify=y_known,
    )

    X_test = X_known.iloc[test_positions].reset_index(drop=True)
    y_test = y_known.iloc[test_positions].reset_index(drop=True)
    labels_test = labels_known.iloc[test_positions].reset_index(drop=True)
    families_test = families_known.iloc[test_positions].reset_index(drop=True)

    # Use the saved model without retraining or changing thresholds.
    Xt = prep.transform(X_test)
    components = detector.predict_components(Xt)

    supervised_pred = np.asarray(
        components["supervised_alert"], dtype=int
    )
    anomaly_pred = np.asarray(
        components["anomaly_alert"], dtype=int
    )
    hybrid_pred = np.asarray(
        components["hybrid_alert"], dtype=int
    )

    result = pd.DataFrame({
        "original_label": labels_test,
        "attack_family": families_test,
        "true_label": y_test,
        "supervised_score": components["supervised_score"],
        "supervised_threshold": detector.supervised_threshold,
        "supervised_prediction": supervised_pred,
        "anomaly_score": components["anomaly_score"],
        "anomaly_threshold": detector.anomaly_threshold,
        "anomaly_prediction": anomaly_pred,
        "hybrid_prediction": hybrid_pred,
    })

    result["hybrid_error_type"] = np.select(
        [
            (result["true_label"] == 0)
            & (result["hybrid_prediction"] == 1),
            (result["true_label"] == 1)
            & (result["hybrid_prediction"] == 0),
        ],
        ["FALSE_POSITIVE", "FALSE_NEGATIVE"],
        default="CORRECT",
    )

    result["supervised_error_type"] = np.select(
        [
            (result["true_label"] == 0)
            & (result["supervised_prediction"] == 1),
            (result["true_label"] == 1)
            & (result["supervised_prediction"] == 0),
        ],
        ["FALSE_POSITIVE", "FALSE_NEGATIVE"],
        default="CORRECT",
    )

    # Save all test predictions and the hybrid errors.
    result.to_csv(
        OUT_DIR / "known_family_test_predictions.csv",
        index=False,
    )

    errors = result[
        result["hybrid_error_type"] != "CORRECT"
    ].copy()
    errors.to_csv(OUT_DIR / "hybrid_errors.csv", index=False)

    summary = (
        errors.groupby(
            [
                "original_label",
                "attack_family",
                "hybrid_error_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["hybrid_error_type", "count"],
            ascending=[True, False],
        )
    )
    summary.to_csv(
        OUT_DIR / "hybrid_error_summary.csv",
        index=False,
    )

    # Save confusion matrices for all three detectors.
    matrices = {}

    for name, pred in [
        ("supervised", supervised_pred),
        ("anomaly", anomaly_pred),
        ("hybrid", hybrid_pred),
    ]:
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        matrices[name] = {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    metadata = {
        "model_path": str(MODEL_PATH),
        "data_dir": str(DATA_DIR),
        "heldout_family_excluded": str(cfg["heldout_family"]),
        "test_rows": int(len(result)),
        "hybrid_error_rows": int(len(errors)),
        "hybrid_false_positives": int(
            (result["hybrid_error_type"] == "FALSE_POSITIVE").sum()
        ),
        "hybrid_false_negatives": int(
            (result["hybrid_error_type"] == "FALSE_NEGATIVE").sum()
        ),
        "confusion_matrices": matrices,
        "note": (
            "Known-family test errors only. The held-out family is "
            "excluded. The saved model and thresholds were not modified."
        ),
    }

    (OUT_DIR / "error_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n=== CONFUSION MATRICES ===")
    print(json.dumps(matrices, indent=2))

    print("\n=== HYBRID ERROR COUNTS ===")
    print("Test rows:", len(result))
    print("False positives:", metadata["hybrid_false_positives"])
    print("False negatives:", metadata["hybrid_false_negatives"])

    print("\n=== HYBRID ERRORS BY LABEL ===")
    if summary.empty:
        print("No hybrid errors found in this test split.")
    else:
        print(summary.to_string(index=False))

    print("\nSaved files in:", OUT_DIR)


if __name__ == "__main__":
    main()
