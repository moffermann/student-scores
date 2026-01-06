# Hyperparameter Tuning Strategy

**Branch:** `hyperparameter-tuning`
**Fecha:** 2026-01-05
**Modelo:** XGBoost Regressor con Optuna

## Descripción

Optimización de hiperparámetros usando Optuna con TPE Sampler (Tree-structured Parzen Estimator) para búsqueda bayesiana. Se mantiene el mismo pipeline de features que el baseline.

## Pipeline

1. Carga de datos (train: 630,000 filas, test: 270,000 filas)
2. Label Encoding para variables categóricas
3. Optimización con Optuna (50 trials, 5-fold CV)
4. Entrenamiento con mejores hiperparámetros
5. Clipping de predicciones al rango [0, 100]
6. Generación de submission

## Features

Sin cambios respecto al baseline:
- 7 features categóricas (Label Encoded)
- 4 features numéricas

**Total features:** 11

## Espacio de Búsqueda

```python
{
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),       # log scale
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 10),
    'reg_alpha': (1e-8, 10.0),          # log scale
    'reg_lambda': (1e-8, 10.0),         # log scale
    'gamma': (1e-8, 1.0)                # log scale
}
```

## Mejores Hiperparámetros Encontrados

```python
best_params = {
    'n_estimators': 485,
    'max_depth': 7,
    'learning_rate': 0.061553,
    'subsample': 0.897410,
    'colsample_bytree': 0.817250,
    'min_child_weight': 10,
    'reg_alpha': 0.003965,
    'reg_lambda': 0.000068,
    'gamma': 0.258847,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}
```

## Comparación con Baseline

| Parámetro | Baseline | Tuned |
|-----------|----------|-------|
| n_estimators | 200 | 485 |
| max_depth | 6 | 7 |
| learning_rate | 0.1 | 0.0616 |
| subsample | 0.8 | 0.897 |
| colsample_bytree | 0.8 | 0.817 |
| min_child_weight | 1 (default) | 10 |
| reg_alpha | 0 (default) | 0.004 |
| reg_lambda | 1 (default) | 0.00007 |
| gamma | 0 (default) | 0.259 |

## Resultados

### Optimización Optuna
| Métrica | Valor |
|---------|-------|
| Trials | 50 |
| Best Trial | 48 |
| Tiempo total | ~9 minutos |

### Cross Validation (5-Fold)
| Fold | RMSE |
|------|------|
| 1 | 8.74980 |
| 2 | 8.75623 |
| 3 | 8.74696 |
| 4 | 8.77023 |
| 5 | 8.78180 |
| **Mean** | **8.76100** |
| **Std** | 0.01314 |

### Predicciones (con clipping)
| Métrica | Valor |
|---------|-------|
| Min | 13.63 |
| Mean | 62.52 |
| Max | 100.00 |

### Mejora vs Baseline
| Métrica | Baseline | Tuned | Diferencia |
|---------|----------|-------|------------|
| CV RMSE | 8.78256 | 8.76100 | -0.02156 |
| Gap to top | 0.243 | 0.221 | -0.022 |

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | study_hours | 0.5873 |
| 2 | sleep_quality | 0.1316 |
| 3 | class_attendance | 0.0870 |
| 4 | study_method | 0.0859 |
| 5 | facility_rating | 0.0763 |
| 6 | sleep_hours | 0.0210 |
| 7 | age | 0.0023 |
| 8 | course | 0.0023 |
| 9 | gender | 0.0022 |
| 10 | internet_access | 0.0021 |
| 11 | exam_difficulty | 0.0020 |

## Observaciones

1. **Mejora modesta**: 0.02156 de reducción en RMSE
2. **Learning rate menor**: El modelo prefiere aprender más lento (0.06 vs 0.1)
3. **Más árboles**: 485 vs 200 para compensar el learning rate menor
4. **Mayor regularización**: min_child_weight=10 y gamma=0.26 previenen overfitting
5. **Clipping aplicado**: Predicciones limitadas a [0, 100]
6. **study_hours sigue dominando**: 58.7% de importancia

## Archivos

- `hyperparameter_tuning.py` - Código de optimización
- `submissions/submission_tuned_20260105_211126.csv` - Submission generada
