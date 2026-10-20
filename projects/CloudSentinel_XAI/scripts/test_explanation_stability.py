
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from src.data.loader import read_flow_csvs
from src.data.preprocessing import clean_frame
from src.explainability import global_permutation_sensitivity


SEED = 42
MODEL_PATH = Path(
    "outputs/experiment_02/artifacts/cloudsentinel_pipeline.joblib"
)
DATA_DIR = Path("data/deduplicated")
CONFIG_PATH = Path("configs/experiment_02.yaml")
OUT_DIR = Path("outputs/experiment_02/explanations")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    prep = bundle["preprocessor"]
    detector = bundle["detector"]
    names = prep.get_feature_names_out().tolist()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    family_map = yaml.safe_load(
        Path("configs/attack_family_map.yaml").read_text()
    )["family_by_label"]

    raw, _ = read_flow_csvs(str(DATA_DIR))
    X, labels = clean_frame(raw)

    X = X.loc[:, list(prep.feature_names_in_)].reset_index(drop=True)
    labels = labels.astype(str).str.strip().reset_index(drop=True)

    mapping = {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in family_map.items()
    }
    families = labels.str.upper().map(mapping)

    if families.isna().any():
        raise ValueError("Some labels are missing from the family map.")

    benign = labels.str.upper().eq(
        str(cfg.get("benign_label", "BENIGN")).upper()
    )
    heldout = families.eq(str(cfg["heldout_family"]).strip().upper())

    # Keep the held-out family excluded from stability analysis.
    known = ~heldout
    X_known = X.loc[known].reset_index(drop=True)
    y_known = (~benign.loc[known]).astype(int).reset_index(drop=True)

    X_trainval, _, y_trainval, _ = train_test_split(
        X_known,
        y_known,
        test_size=float(cfg["test_size"]),
        random_state=SEED,
        stratify=y_known,
    )

    validation_fraction = (
        float(cfg["validation_size"]) / (1.0 - float(cfg["test_size"]))
    )

    _, X_val, _, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=validation_fraction,
        random_state=SEED,
        stratify=y_trainval,
    )

    # Compare multiple bounded subsets of the same validation set.
    # A different seed changes the sampled rows, not the trained model.
    results = []

    for subset_seed in [11, 42, 73, 104, 205]:
        n = min(1000, len(X_val))
        X_sub = X_val.sample(n=n, random_state=subset_seed)
        y_sub = y_val.loc[X_sub.index].to_numpy()

        X_enc = prep.transform(X_sub)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        X_enc = np.asarray(X_enc, dtype=float)

        scores = detector.supervised.predict_proba(X_enc)[:, 1]
        baseline_ap = average_precision_score(y_sub, scores)

        ranked = global_permutation_sensitivity(
            detector.supervised,
            X_enc,
            y_sub,
            names,
            n_repeats=3,
            seed=subset_seed,
            max_features=25,
        )

        ranked["subset_seed"] = subset_seed
        ranked["baseline_pr_auc_check"] = baseline_ap
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        results.append(ranked)

        print(
            f"Seed {subset_seed}: "
            f"rows={n}, PR-AUC={baseline_ap:.6f}, "
            f"top feature={ranked.iloc[0]['feature']}"
        )

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(
        OUT_DIR / "permutation_stability_by_seed.csv", index=False
    )

    # Summarize feature rank consistency across subsets.
    summary = (
        all_results.groupby("feature")
        .agg(
            appearances=("rank", "count"),
            mean_rank=("rank", "mean"),
            rank_std=("rank", "std"),
            mean_pr_auc_drop=("mean_pr_auc_drop", "mean"),
            std_pr_auc_drop=("mean_pr_auc_drop", "std"),
        )
        .sort_values(
            ["appearances", "mean_rank"],
            ascending=[False, True],
        )
        .reset_index()
    )

    summary.to_csv(
        OUT_DIR / "permutation_stability_summary.csv", index=False
    )

    print("\n=== TOP STABILITY SUMMARY ===")
    print(summary.head(15).to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "permutation_stability_by_seed.csv")
    print(OUT_DIR / "permutation_stability_summary.csv")
    print(
        "\nInterpretation: repeated subset analysis measures ranking "
        "consistency on this validation set; it does not establish "
        "generalization to other datasets."
    )


if __name__ == "__main__":
    main()
