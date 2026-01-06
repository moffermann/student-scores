# Baseline Strategy

**Branch:** `baseline`
**Fecha:** 2026-01-05
**Modelo:** XGBoost Regressor

## Descripción

Modelo baseline simple usando XGBoost con Label Encoding para variables categóricas. Sin feature engineering adicional.

## Pipeline

1. Carga de datos (train: 630,000 filas, test: 270,000 filas)
2. Label Encoding para variables categóricas
3. Entrenamiento con XGBoost
4. 5-Fold Cross Validation
5. Predicción y generación de submission

## Features

### Categóricas (7) - Label Encoded
| Feature | Valores únicos |
|---------|----------------|
| gender | 3 |
| course | 7 |
| internet_access | 2 |
| sleep_quality | 3 |
| study_method | 5 |
| facility_rating | 3 |
| exam_difficulty | 3 |

### Numéricas (4) - Sin transformación
| Feature | Rango |
|---------|-------|
| age | 17 - 24 |
| study_hours | continuo |
| class_attendance | continuo |
| sleep_hours | 4.1 - 9.9 |

**Total features:** 11

## Hiperparámetros

```python
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}
```

## Resultados

### Cross Validation (5-Fold)
| Fold | RMSE |
|------|------|
| 1 | 8.77151 |
| 2 | 8.77918 |
| 3 | 8.76855 |
| 4 | 8.78984 |
| 5 | 8.80373 |
| **Mean** | **8.78256** |
| **Std** | 0.01289 |

### Predicciones
| Métrica | Valor |
|---------|-------|
| Min | 14.24 |
| Mean | 62.52 |
| Max | 104.93 |

### Comparación con Leaderboard
| Métrica | Valor |
|---------|-------|
| CV RMSE | 8.78256 |
| Top Leaderboard | ~8.54 |
| Gap | 0.243 |

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | study_hours | 0.5707 |
| 2 | sleep_quality | 0.1135 |
| 3 | study_method | 0.1063 |
| 4 | class_attendance | 0.0966 |
| 5 | facility_rating | 0.0767 |
| 6 | sleep_hours | 0.0291 |
| 7 | age | 0.0016 |
| 8 | course | 0.0016 |
| 9 | gender | 0.0015 |
| 10 | internet_access | 0.0012 |
| 11 | exam_difficulty | 0.0011 |

## Observaciones

1. **study_hours** domina con 57% de importancia
2. Las predicciones exceden el rango válido del target (19.6 - 100)
3. Variables como `exam_difficulty`, `internet_access`, `gender` tienen muy poca importancia
4. Potencial mejora con clipping de predicciones al rango válido

## Archivos

- `baseline.py` - Código del modelo
- `submissions/submission_baseline_20260105_203254.csv` - Primera submission
