# Playground Series S6E1: Predicting Student Test Scores

## Estado del Proyecto

**Fecha**: 5 Enero 2026
**Competencia**: https://www.kaggle.com/competitions/playground-series-s6e1
**Deadline**: 31 Enero 2026
**Premio**: Swag

## Resultados Actuales

| Modelo | CV RMSE | LB Score |
|--------|---------|----------|
| Baseline XGBoost | 8.783 | **8.755** |
| Top Leaderboard | - | 8.542 |

**Gap al top**: 0.213

## Dataset

- **Train**: 630,000 filas, 13 columnas
- **Test**: 270,000 filas, 12 columnas
- **Target**: `exam_score` (regresión, rango 19.6-100)
- **Missing values**: 0

### Features

**Numéricas (4)**:
- age (17-24)
- study_hours
- class_attendance
- sleep_hours (4.1-9.9)

**Categóricas (7)**:
- gender: male, female, other
- course: b.tech, b.sc, b.com, bca, bba, ba, b.arch
- internet_access: yes, no
- sleep_quality: poor, good, average
- study_method: coaching, self-study, mixed, group study, online videos
- facility_rating: low, medium, high
- exam_difficulty: easy, moderate, hard

### Feature Importance (Baseline)

| Feature | Importance |
|---------|------------|
| **study_hours** | 57% |
| sleep_quality | 11% |
| study_method | 11% |
| class_attendance | 10% |
| facility_rating | 8% |

## Archivos

```
/home/mauro/kaggle/student-scores/
├── data/
│   ├── train.csv (46 MB)
│   ├── test.csv (18 MB)
│   └── sample_submission.csv
├── notebooks/
├── src/
├── submissions/
│   └── submission_baseline_20260105_203254.csv
├── baseline.py
└── claude.md (este archivo)
```

## Próximos Pasos

1. **Feature Engineering**:
   - Interacciones: study_hours × sleep_quality, study_hours × class_attendance
   - Binning de variables numéricas
   - Target encoding para categóricas
   - Polynomial features

2. **Hyperparameter Tuning**:
   - Optuna con 100+ trials
   - Tunear: max_depth, learning_rate, n_estimators, subsample, colsample_bytree

3. **Modelos Adicionales**:
   - LightGBM
   - CatBoost
   - Ensemble/Stacking

## Código Baseline

```python
# XGBoost parameters usados
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

## Notas

- LB score es mejor que CV (8.755 vs 8.783) - no hay overfitting
- `study_hours` domina el modelo (57% importancia)
- Métrica: RMSE (menor es mejor)
