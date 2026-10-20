# CloudSentinel-XAI experiment report

- Rows loaded: 70,566
- Held-out family: DOS (20,000 rows)

## known_family_test

| Detector | Precision | Recall | F1 | FPR | PR-AUC |
|---|---:|---:|---:|---:|---:|
| supervised | 0.9990 | 0.9988 | 0.9989 | 0.0007 | 1.0000 |
| anomaly_only | 0.4364 | 0.0117 | 0.0227 | 0.0103 | 0.5386 |
| hybrid_union | 0.9844 | 0.9988 | 0.9916 | 0.0108 | 0.9997 |
## heldout_family_test_plus_benign

| Detector | Precision | Recall | F1 | FPR | PR-AUC |
|---|---:|---:|---:|---:|---:|
| supervised | 0.9948 | 0.0383 | 0.0738 | 0.0007 | 0.9875 |
| anomaly_only | 0.9853 | 0.2071 | 0.3423 | 0.0103 | 0.9165 |
| hybrid_union | 0.9869 | 0.2454 | 0.3931 | 0.0108 | 0.9268 |

## Limitations

Held-out family is a novelty proxy, not proof of detecting all real-world zero-days. Hybrid union can increase false positives. Verify mapping and dataset provenance.