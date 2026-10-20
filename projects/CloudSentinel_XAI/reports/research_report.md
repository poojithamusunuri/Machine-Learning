# CloudSentinel-XAI
## Explainable Hybrid Machine Learning for Network Intrusion Detection

**Project type:** Research and experimental prototype

**Dataset:** CSE-CIC-IDS2018

**Primary objective:** Evaluate a supervised intrusion detector combined with benign-only anomaly detection, and investigate feature sensitivity and detection performance on a held-out attack family.

---

## 1. Abstract

This project investigates a hybrid machine-learning approach for network intrusion detection that combines a supervised Random Forest classifier with an Isolation Forest trained on benign traffic. A held-out attack-family protocol is used to evaluate how the detectors behave on a family excluded from model training and threshold calibration. Feature-sensitivity and permutation-based analyses are used to investigate the behavior of the supervised classifier. Experiments are conducted using selected CSE-CIC-IDS2018 traffic CSV files and a deduplicated development sample. Results are interpreted in the context of dataset limitations, class separability, false alarms, missed attacks, and the distinction between offline held-out-family evaluation and real-world zero-day detection.

## 2. Problem Statement

Supervised intrusion detection models can perform well on attack families represented during training but may perform less effectively on attack patterns that differ from their training data. Anomaly detection offers a complementary approach by identifying traffic that differs from a benign reference distribution, although it may also flag legitimate unusual traffic.

This project evaluates whether combining supervised classification and benign-only anomaly detection provides a useful trade-off between known-family detection and held-out-family detection.

## 3. Objectives

1. Prepare and audit network-flow data.
2. Train a supervised Random Forest classifier.
3. Train an Isolation Forest using benign training traffic only.
4. Evaluate a hybrid detector that combines their alerts.
5. Evaluate performance on a held-out attack family.
6. investigate local and global feature sensitivity.
7. Measure the consistency of feature rankings across validation subsets.
8. Document false positives, false negatives, reproducibility constraints, and limitations.

## 4. Dataset and Data Preparation

The experiments use selected CSV files from CSE-CIC-IDS2018. The development dataset was sampled from the source files and then deduplicated within each source file based on flow features, excluding the label.

The initial sample contained 70,566 rows. Deduplication removed 3,596 repeated rows, leaving 66,970 rows for Experiment 02.

The deduplicated dataset contains benign traffic and multiple attack labels. Labels are mapped to broader attack families using `configs/attack_family_map.yaml`.

The source files, label distributions, missing values, infinite values, and repeated feature rows were audited. Data cleaning excludes labels and identifying flow metadata from the model features and handles missing values through the fitted preprocessing pipeline.

## 5. Methodology

### 5.1 Supervised detector

A Random Forest classifier with 300 trees and balanced subsampling class weights is trained using known-family training data.

### 5.2 Anomaly detector

An Isolation Forest with 250 trees is fitted using benign training rows only. Its anomaly threshold is calibrated using benign validation scores to target a 1% false-positive rate.

### 5.3 Hybrid detector

The hybrid detector raises an alert if either the supervised classifier or the anomaly detector raises an alert.

### 5.4 Evaluation protocol

The data is split into training, validation, and test partitions. The selected held-out attack family is excluded from training and threshold calibration. Its evaluation combines held-out-family attack examples with benign examples.

The supervised threshold is selected using validation F1, while the anomaly threshold is calibrated against benign validation scores. Reported test results are evaluated separately from threshold selection.

## 6. Experiment 01: Initial Sample

The initial sampled dataset contained 70,566 rows. DOS was selected as the held-out family.

On the known-family test split, the supervised detector achieved an F1-score of approximately 0.9989, while the hybrid detector achieved approximately 0.9916.

On the held-out DOS evaluation, the hybrid detector achieved recall of approximately 0.2454 and an F1-score of approximately 0.3931.

These results demonstrate strong performance on known-family traffic but substantially lower recall on the held-out family.

## 7. Experiment 02: Deduplicated Sample

The deduplicated dataset contained 66,970 rows, with 3,596 repeated rows removed.

On the known-family test split, the supervised detector achieved an F1-score of approximately 0.9980, while the hybrid detector achieved approximately 0.9893.

On the held-out DOS evaluation, the hybrid detector achieved recall of approximately 0.2304 and an F1-score of approximately 0.3734.

The observed metrics were slightly lower than those of Experiment 01. This comparison is a sensitivity analysis, not proof that deduplication alone caused the change, because the dataset size and class distributions also changed.

## Error Analysis

### Known-Family Test-Set Analysis

Error analysis was conducted on the 9,520-record known-family test set from Experiment 02. The test set contained 6,000 benign records and 3,520 attack records. The hybrid detector produced 5,928 true negatives, 72 false positives, 4 false negatives, and 3,516 true positives. This corresponds to an attack recall of approximately 99.89% and a false-positive rate of 1.20%.

## Held-Out DoS Error Analysis

### Evaluation Setup

The held-out-family analysis evaluated the saved Experiment 02 model on 49,371 records from the deduplicated dataset. The evaluation contained 19,372 DoS attack records and 29,999 benign records. The DoS family was excluded from supervised training, and the existing model and decision thresholds were used without retraining or tuning on the evaluation records.

### Detection Results

The supervised Random Forest produced 309 true positives and 19,063 false negatives, corresponding to a DoS recall of approximately 1.60%. The Isolation Forest detected 4,155 DoS records, with 15,217 false negatives and 285 false positives. The hybrid detector produced 4,464 true positives, 14,908 false negatives, 322 false positives, and 29,677 true negatives.

The hybrid detector achieved a DoS recall of approximately 23.04%, compared with 21.45% for the Isolation Forest and 1.60% for the supervised model. Thus, the hybrid improved recall over the anomaly detector alone by approximately 1.60 percentage points, while generating 37 additional false positives.

### Error Distribution by Attack Label

The hybrid error summary identified 9,981 false negatives labelled `DoS attacks-GoldenEye` and 4,927 labelled `DoS attacks-Slowloris`. Together, these account for 14,908 missed DoS records. The detector also generated 322 false positives on benign traffic.

These results indicate that the held-out DoS family was substantially harder to detect than the known attack families in the known-family test set. Although the anomaly detector contributes useful detection capability beyond supervised classification, the hybrid system still misses approximately 76.96% of the held-out DoS records.

### Interpretation and Limitations

The results demonstrate partial generalization to a held-out attack family, not reliable detection of all unseen attacks. The anomaly detector can identify some DoS flows as anomalous relative to its learned benign baseline, but this capability is limited at the selected operating threshold.

The held-out evaluation includes benign traffic from the deduplicated dataset, whereas the earlier held-out metrics used a smaller benign evaluation subset. Consequently, false-positive counts should not be compared directly without accounting for the different evaluation populations.

The dataset also contains potential capture-day confounding: the held-out DoS records originate from a different capture day than the known-family training data. Therefore, the observed performance cannot be attributed solely to attack-family novelty. Further evaluation across capture days, additional attack families, and independent traffic sources is required before making broader claims about zero-day detection.


### False-Positive Attribution

The supervised Random Forest generated 10 false positives, while the Isolation Forest generated 62 false positives on benign traffic. No benign record was incorrectly flagged by both components in this test set. Under the hybrid system's OR-based decision rule, the false-positive counts therefore combined to produce 72 hybrid false positives.

This finding illustrates a trade-off in the hybrid design. Combining supervised classification with anomaly detection can provide an additional detection mechanism, but an OR-based rule also allows an anomaly alert to trigger a final alert even when the supervised model considers the record benign. In this experiment, the anomaly detector contributed more false positives than the supervised classifier.

### False-Negative Analysis

The hybrid detector missed four attack records: three labelled `Brute Force -Web` and one labelled `Brute Force -XSS`. All four records had supervised attack scores below the selected threshold of 0.350573 and anomaly scores below the anomaly threshold of 0.638998. Neither component therefore triggered an alert for these records.

The observed errors suggest that these particular web-attack records were not sufficiently distinguishable from benign traffic under the trained models and selected thresholds. Further investigation of their flow characteristics and comparison with correctly detected examples would be required to identify the underlying reasons.

### Implications for the Hybrid Design

The results show that the hybrid detector achieved high recall on the known-family test set, while introducing additional false positives compared with the supervised model alone. This reflects the sensitivity of an OR-based hybrid system to the false-positive behaviour of either component.

Future work should evaluate alternative fusion strategies, threshold-selection methods, and false-positive reduction techniques. Any revised configuration should be assessed on held-out validation data and then evaluated on an untouched test set to avoid tuning directly to the reported test errors.

These findings are limited to the sampled, deduplicated dataset and the experimental split used here. They should not be interpreted as proof of equivalent performance on live network traffic or genuinely novel attacks.

### Attack-Specific Detection and Component Agreement

The held-out DoS evaluation revealed substantial differences between GoldenEye and Slowloris detection. For GoldenEye, the supervised model detected 2 of 9,997 records, the Isolation Forest detected 14, and the hybrid detected 16, corresponding to recalls of approximately 0.02%, 0.14%, and 0.16%, respectively.

For Slowloris, the supervised model detected 307 of 9,375 records, while the Isolation Forest detected 4,141. The hybrid detected 4,448 records, achieving a recall of approximately 47.45%, compared with 44.17% for the Isolation Forest and 3.27% for the supervised model.

Component-agreement analysis showed that the supervised and anomaly components did not jointly detect any held-out attack records at the selected thresholds. The supervised component uniquely detected 309 records, the anomaly component uniquely detected 4,155 records, and both components missed 14,908 records.

These findings demonstrate that the anomaly detector's contribution depends strongly on the held-out attack type. The hybrid improves detection by combining complementary predictions, but its OR-based fusion cannot detect attacks missed by both components. The poor GoldenEye recall also shows that performance on one held-out attack family cannot be assumed to generalize to another.

The results should be interpreted within the limits of the dataset, capture conditions, sampling strategy, and selected thresholds. Further evaluation on independent traffic sources and additional held-out attack families is needed before making broader claims about zero-day detection.


## 8. Explainability Analysis

### 8.1 Local feature sensitivity

For one known-family validation attack, the supervised model's baseline attack score was approximately 0.999936.

Replacing `Fwd Seg Size Min` with its background median reduced the score to approximately 0.505847. This indicates high sensitivity to that feature for the selected example.

### 8.2 Global permutation sensitivity

The initial global permutation analysis reported a baseline validation PR-AUC of 1.0 on its 1,000-row subset. The largest mean PR-AUC drops were approximately 0.001163 for `Init Fwd Win Byts` and 0.000524 for `Fwd Seg Size Min`.

### 8.3 Within-validation stability

Five 1,000-row subsets were evaluated using different sampling seeds. `Fwd Seg Size Min` had a mean rank of 1.4, and `Init Fwd Win Byts` had a mean rank of 1.6. Both features ranked first or second in all five subsets.

The full known-family validation set had PR-AUC of approximately 0.999934 and ROC-AUC of approximately 0.999958.

These results indicate ranking consistency within the selected validation split. They do not establish stability across independent datasets.

## 9. Limitations

1. The experiments use selected files and a sampled dataset rather than the complete benchmark.
2. Held-out DOS examples come from a different capture day from several known-family examples, so capture-day effects may confound the results.
3. A held-out attack family is a proxy for novelty evaluation; it is not proof of real-world zero-day detection.
4. High known-family scores may reflect strong separability in this dataset and should not be assumed to generalize to operational networks.
5. Local median replacement and feature permutation are sensitivity diagnostics, not causal explanations.
6. Correlated features can mask each other's contributions, and independently permuting a feature can produce unrealistic flow combinations.
7. The five stability subsets come from the same validation split.
8. Independent-dataset testing, more comprehensive error analysis, and further explanation-fidelity checks remain necessary.

## 10. Conclusion

The experiments show that the supervised detector performs strongly on known-family test traffic, while held-out DOS detection remains substantially more difficult. The hybrid approach increases held-out-family recall compared with the supervised detector alone, at the cost of additional false positives.

The explainability analysis identifies `Fwd Seg Size Min` and `Init Fwd Win Byts` as consistently high-ranking features within the validation data. However, the small global PR-AUC drops indicate that these rankings should be interpreted cautiously.

Further work should investigate false positives and missed attacks, test across independent capture days, and assess whether the observed detection and explanation behavior generalizes beyond this sampled benchmark.

## 11. Reproducibility and Artifacts

- Experiment configuration: `configs/experiment.yaml`
- Deduplicated experiment configuration: `configs/experiment_02.yaml`
- Attack-family mapping: `configs/attack_family_map.yaml`
- Experiment 01 artifacts: `outputs/experiment_01/`
- Experiment 02 artifacts: `outputs/experiment_02/`
- Explainability outputs: `outputs/experiment_02/explanations/`
- Experiment runner: `src/experiments/run_experiment.py`
- Explainability script: `scripts/run_explainability.py`
- Stability script: `scripts/test_explanation_stability.py`

The report should be updated when additional experiments and error analysis are completed. All reported metrics should be traceable to their saved experiment outputs.
