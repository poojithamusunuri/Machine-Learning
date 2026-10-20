# Research protocol
1. Download official files and record names, sizes, retrieval date and SHA-256.
2. Audit label spelling, class counts, missing/infinite values and duplicates.
3. Freeze label-family mapping and held-out family before fitting.
4. Exclude held-out family from training, validation, feature selection and threshold calibration.
5. Fit preprocessing only on training data; train Random Forest on benign plus known attacks and Isolation Forest on benign training flows only.
6. Calibrate thresholds on validation data, never held-out rows.
7. Compare supervised, anomaly-only and hybrid on known-family and held-out-family tests.
8. Report precision, recall, F1, PR-AUC, FPR, per-family metrics and error examples. Repeat for multiple held-out families when sample sizes allow.

Validity risks: label spelling can invalidate holdout; random row splits may leak correlated flows; feature attribution is not causal; flow data are not a live cloud sensor; union logic can increase false alerts.
