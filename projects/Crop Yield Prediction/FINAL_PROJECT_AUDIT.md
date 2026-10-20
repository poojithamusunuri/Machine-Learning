# Crop Yield Prediction — Final Audit

## Status
All 11 supplied project notebooks were replaced with a consistent, leakage-controlled workflow and checked for JSON validity, Windows-only paths, model artifact loading, and recorded outputs.

## Dataset facts verified from the supplied CSV
- Initial dataset: 19,689 rows × 13 columns.
- Target: `Yield` (continuous regression target).
- Missing values: 2 `Annual_Rainfall`, 2 `Fertilizer`, 2 `Pesticide`, 1 `Yield`, 2 `Avg_Temperature` = 9 missing cells.
- Duplicate rows: 0.
- The single missing target row is removed; predictor missingness is handled inside preprocessing pipelines.
- `Production` is excluded from model inputs to prevent target leakage.

## Preprocessing corrections
- Train/test split is performed before fitting imputers, encoders, or scalers.
- Unknown categorical values are handled safely.
- Two input-only engineered features are used: `Fertilizer_per_Area` and `Pesticide_per_Area`.
- The saved model artifacts are complete pipelines, keeping preprocessing with the estimator.
- No Windows `C:\...` paths remain in the corrected notebooks.

## Validation and testing
- Test split: 20%, `random_state=42`.
- Model selection validation: 3-fold shuffled CV on the training split.
- Hyperparameter tuning: 4 randomized XGBoost configurations with 2-fold CV, using only the training split.
- The final test set is not used for hyperparameter selection.

## Final measured results
| Model | MAE | RMSE | R² | Mean CV RMSE |
|---|---:|---:|---:|---:|
| XGBoost (Tuned) | 12.9041 | 214.9211 | 0.9346 | 155.8302 |
| Random Forest | 11.8115 | 193.7821 | 0.9469 | 169.3937 |
| XGBoost | 14.5846 | 164.5191 | 0.9617 | 182.9476 |
| SVR (5k training subset) | 68.4255 | 821.5300 | 0.0448 | — |

**Selection rule:** the tuned XGBoost pipeline is selected because it has the lowest mean 3-fold validation RMSE (155.8302). The untouched test result is reported transparently and is not cherry-picked for model selection.

## SVR note
Full RBF SVR on the complete training set is computationally expensive for this dataset. The corrected notebook therefore uses a fixed, reproducible 5,000-row subset of the training split and explicitly reports this limitation. The full test set is still used for its evaluation.

## Final demonstration
The selected tuned XGBoost pipeline predicts a sample unseen input with an output of approximately **11.500705** in the dataset's Yield units. `Yield` and `Production` are not supplied to the model during prediction.
