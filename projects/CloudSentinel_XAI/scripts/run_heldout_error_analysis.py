
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import confusion_matrix

from src.data.loader import read_flow_csvs
from src.data.preprocessing import clean_frame


CONFIG_PATH = Path("configs/experiment_02.yaml")
MODEL_PATH = Path(
    "outputs/experiment_02/artifacts/cloudsentinel_pipeline.joblib"
)
DATA_DIR = Path("data/deduplicated")
OUT_DIR = Path("outputs/experiment_02/heldout_error_analysis")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    prep = bundle["preprocessor"]
    detector = bundle["detector"]
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    family_map = yaml.safe_load(
        Path("configs/attack_family_map.yaml").read_text()
    )["family_by_label"]

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

    benign_name = str(cfg.get("benign_label", "BENIGN")).upper()
    heldout_name = str(cfg["heldout_family"]).strip().upper()

    benign = labels.str.upper().eq(benign_name)
    heldout = families.eq(heldout_name)

    # Evaluate only held-out-family attacks and benign traffic.
    # Other known attack families are excluded from this evaluation.
    evaluation_mask = heldout | benign

    X_eval = X.loc[evaluation_mask].reset_index(drop=True)
    labels_eval = labels.loc[evaluation_mask].reset_index(drop=True)
    families_eval = families.loc[evaluation_mask].reset_index(drop=True)
    y_eval = (
        ~labels_eval.str.upper().eq(benign_name)
    ).astype(int).reset_index(drop=True)

    if not heldout.any():
        raise ValueError(
            f"No records found for held-out family {heldout_name}."
        )

    # Use the saved preprocessor and detector without retraining.
    Xt = prep.transform(X_eval)
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
        "original_label": labels_eval,
        "attack_family": families_eval,
        "true_label": y_eval,
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

    result["anomaly_error_type"] = np.select(
        [
            (result["true_label"] == 0)
            & (result["anomaly_prediction"] == 1),
            (result["true_label"] == 1)
            & (result["anomaly_prediction"] == 0),
        ],
        ["FALSE_POSITIVE", "FALSE_NEGATIVE"],
        default="CORRECT",
    )

    result.to_csv(
        OUT_DIR / "heldout_test_predictions.csv", index=False
    )

    errors = result[
        result["hybrid_error_type"] != "CORRECT"
    ].copy()
    errors.to_csv(
        OUT_DIR / "heldout_hybrid_errors.csv", index=False
    )

    summary = (
        errors.groupby(
            ["original_label", "attack_family", "hybrid_error_type"],
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
        OUT_DIR / "heldout_hybrid_error_summary.csv", index=False
    )

    matrices = {}
    for name, pred in [
        ("supervised", supervised_pred),
        ("anomaly", anomaly_pred),
        ("hybrid", hybrid_pred),
    ]:
        tn, fp, fn, tp = confusion_matrix(
            y_eval, pred, labels=[0, 1]
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
        "heldout_family": heldout_name,
        "evaluation_rows": int(len(result)),
        "heldout_attack_rows": int(heldout.sum()),
        "benign_rows": int(benign.sum()),
        "hybrid_error_rows": int(len(errors)),
        "confusion_matrices": matrices,
        "note": (
            "Held-out-family evaluation on the existing deduplicated "
            "dataset using the saved model and thresholds. No retraining "
            "or threshold tuning was performed. This is a held-out "
            "attack-family proxy, not proof of real-world zero-day "
            "detection."
        ),
    }

    (OUT_DIR / "heldout_error_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n=== HELD-OUT EVALUATION ===")
    print("Held-out family:", heldout_name)
    print("Evaluation rows:", len(result))
    print("Held-out attack rows:", int(heldout.sum()))
    print("Benign rows:", int(benign.sum()))

    print("\n=== CONFUSION MATRICES ===")
    print(json.dumps(matrices, indent=2))

    print("\n=== HYBRID ERRORS BY LABEL ===")
    if summary.empty:
        print("No hybrid errors found.")
    else:
        print(summary.to_string(index=False))

    print("\nSaved files in:", OUT_DIR)


if __name__ == "__main__":
    main()
