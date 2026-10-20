# CloudSentinel-XAI experiment report

- Rows loaded: 66,970
- Held-out family: DOS (19,372 rows)

## known_family_test

| Detector | Precision | Recall | F1 | FPR | PR-AUC |
|---|---:|---:|---:|---:|---:|
| supervised | 0.9972 | 0.9989 | 0.9980 | 0.0017 | 0.9999 |
| anomaly_only | 0.1948 | 0.0043 | 0.0083 | 0.0103 | 0.5180 |
| hybrid_union | 0.9799 | 0.9989 | 0.9893 | 0.0120 | 0.9996 |
## heldout_family_test_plus_benign

| Detector | Precision | Recall | F1 | FPR | PR-AUC |
|---|---:|---:|---:|---:|---:|
| supervised | 0.9687 | 0.0160 | 0.0314 | 0.0017 | 0.9833 |
| anomaly_only | 0.9853 | 0.2145 | 0.3523 | 0.0103 | 0.9164 |
| hybrid_union | 0.9841 | 0.2304 | 0.3734 | 0.0120 | 0.9200 |

## Limitations

Held-out family is a novelty proxy, not proof of detecting all real-world zero-days. Hybrid union can increase false positives. Verify mapping and dataset provenance.