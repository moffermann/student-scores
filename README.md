# Student Exam Score Prediction

Kaggle competition to predict student exam scores based on various features.

## Competition Overview

- **Goal**: Predict `exam_score` (regression, 0-100)
- **Metric**: RMSE (Root Mean Square Error)
- **Train samples**: 630,000
- **Test samples**: 270,000

## Features

### Categorical (7)
| Feature | Values | Description |
|---------|--------|-------------|
| gender | male, female, other | Student gender |
| course | b.com, b.sc, b.tech, ba, bba, bca, diploma | Academic program |
| internet_access | yes, no | Internet availability |
| sleep_quality | good, average, poor | Sleep quality |
| study_method | coaching, group study, mixed, online videos, self-study | Study approach |
| facility_rating | high, medium, low | School facility rating |
| exam_difficulty | easy, moderate, hard | Perceived exam difficulty |

### Numerical (4)
| Feature | Range | Correlation with Target |
|---------|-------|------------------------|
| study_hours | continuous | 0.762 (strong positive) |
| class_attendance | continuous | 0.361 (moderate positive) |
| sleep_hours | continuous | 0.167 (weak positive) |
| age | 17-24 | 0.011 (negligible) |

## Best Model: PL + Meta-Stacking

**CV RMSE: 8.72238**

### Architecture

```
1. Pseudo-Labeling
   - Train 3-model ensemble (XGBoost + CatBoost + HistGB)
   - Generate pseudo-labels for test data
   - Retrain with augmented data (train + pseudo-labeled test)

2. Profile-Weighted Meta-Stacking
   - Create profile clusters from categorical combinations
   - Train Ridge meta-model with cluster indicators
   - Final prediction = meta_model(xgb_pred, cat_pred, hgb_pred, cluster)
```

### Model Parameters

**XGBoost (GPU)**
```python
{
    'n_estimators': 1156,
    'max_depth': 6,
    'learning_rate': 0.059577,
    'subsample': 0.801002,
    'colsample_bytree': 0.992775,
    'min_child_weight': 5,
    'gamma': 0.217669,
    'reg_alpha': 0.254105,
    'reg_lambda': 0.015326,
    'tree_method': 'hist',
    'device': 'cuda'
}
```

**CatBoost (GPU)**
```python
{
    'iterations': 1660,
    'depth': 6,
    'learning_rate': 0.171473,
    'l2_leaf_reg': 5.500127,
    'min_data_in_leaf': 65,
    'random_strength': 2.339920,
    'bagging_temperature': 0.403898,
    'task_type': 'GPU'
}
```

**HistGradientBoosting**
```python
{
    'max_iter': 1371,
    'max_depth': 5,
    'learning_rate': 0.066045,
    'max_leaf_nodes': 55,
    'min_samples_leaf': 20,
    'l2_regularization': 2.912878
}
```

## Experiment History

| # | Strategy | CV RMSE | Improvement |
|---|----------|---------|-------------|
| 1 | Baseline XGBoost | 8.78256 | - |
| 2 | XGBoost Hyperparameter Tuning | 8.76100 | +0.022 |
| 3 | CatBoost Native | 8.76817 | - |
| 4 | CatBoost Tuned (GPU) | 8.74974 | +0.033 |
| 5 | 2-Model Stacking (XGB+CAT) | 8.73771 | +0.045 |
| 6 | 3-Model Stacking (+HistGB) | 8.73339 | +0.049 |
| 7 | Pseudo-Labeling | 8.72348 | +0.059 |
| 8 | **PL + Meta-Stacking** | **8.72238** | **+0.060** |

### Experiments That Did NOT Improve

| Strategy | CV RMSE | Notes |
|----------|---------|-------|
| Feature Engineering | 8.76936 | Trees capture interactions automatically |
| Target Encoding | 8.78715 | CatBoost already handles this |
| Per-Course Models | 8.80795 | Global model generalizes better |
| Per-Gender Models | 8.77330 | Global model generalizes better |
| Custom Neural Network | 8.88980 | Trees outperform for tabular data |
| Hybrid Embeddings + Trees | 8.73686 | Native categorical handling is better |

## Key Insights

1. **Profile Analysis**: 5,668 unique categorical combinations exist, with exam score means ranging from 44 to 84 points
2. **Correlation Variability**: Correlations between features and target vary significantly by profile (e.g., class_attendance: 0.08 to 0.62)
3. **Tree Models Excel**: CatBoost and XGBoost with native categorical handling outperform custom approaches
4. **Pseudo-Labeling Works**: Adding test predictions as training data provides consistent improvement
5. **Ensemble Diversity**: Combining XGBoost, CatBoost, and HistGradientBoosting provides complementary predictions

## Project Structure

```
student-scores/
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
│   └── submission_*.csv
├── docs/
│   └── strategies/
│       ├── hyperparameter-tuning.md
│       ├── stacking-ensemble.md
│       ├── stacking-3models.md
│       ├── pseudo-labeling.md
│       └── outlier-analysis.md
├── baseline.py
├── hyperparameter_tuning.py
├── catboost_native.py
├── catboost_tuning.py
├── stacking_ensemble.py
├── stacking_3models.py
├── pseudo_labeling.py
├── pseudo_labeling_profile_weighted.py  # Best model
├── explore_profiles.py
├── custom_model.py
├── custom_model_v2.py
├── hybrid_embedding_model.py
├── profile_weighted_ensemble.py
└── README.md
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
catboost
torch (for custom models)
optuna (for hyperparameter tuning)
```

## Usage

Run the best model:
```bash
python pseudo_labeling_profile_weighted.py
```

## Author

Mauricio Offermann (mauricio.offermann@gocode.cl)

## License

MIT
