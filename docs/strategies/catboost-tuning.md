# CatBoost Tuning Strategy

**Branch:** `catboost-tuning`
**Fecha:** 2026-01-06
**Modelo:** CatBoost Regressor con Optuna + GPU

## Descripción

Optimización de hiperparámetros de CatBoost usando Optuna con TPE Sampler y entrenamiento en GPU (RTX 5060 Ti). Mantiene categóricas nativas de CatBoost.

## Pipeline

1. Carga de datos
2. Identificación de columnas categóricas (nativas)
3. Optimización con Optuna (50 trials, 5-fold CV, GPU)
4. Entrenamiento final con mejores parámetros
5. Clipping de predicciones
6. Generación de submission

## Features

Sin transformación manual:
- 7 features categóricas (nativas)
- 4 features numéricas

**Total features:** 11

## Espacio de Búsqueda

```python
{
    'iterations': (500, 2000),
    'depth': (4, 10),
    'learning_rate': (0.01, 0.3),        # log scale
    'l2_leaf_reg': (1, 10),
    'min_data_in_leaf': (1, 100),
    'random_strength': (0.1, 10),        # log scale
    'bagging_temperature': (0, 1),
    'border_count': (32, 255)
}
```

## Mejores Hiperparámetros Encontrados

```python
best_params = {
    'iterations': 1660,
    'depth': 6,
    'learning_rate': 0.171473,
    'l2_leaf_reg': 5.500127,
    'min_data_in_leaf': 65,
    'random_strength': 2.339920,
    'bagging_temperature': 0.403898,
    'border_count': 254,
    'task_type': 'GPU',
    'devices': '0'
}
```

## Resultados

### Optimización Optuna
| Métrica | Valor |
|---------|-------|
| Trials | 50 |
| Best Trial | 45 |
| Tiempo total | ~2h 19min |
| Best RMSE (search) | 8.75041 |

### Cross Validation Final (5-Fold)
| Fold | RMSE | Best Iteration |
|------|------|----------------|
| 1 | 8.73949 | 1078 |
| 2 | 8.74488 | 1658 |
| 3 | 8.73644 | 1038 |
| 4 | 8.75750 | 1137 |
| 5 | 8.77039 | 1153 |
| **Mean** | **8.74974** | ~1213 |
| **Std** | 0.01259 | - |

### Comparación con Estrategias Anteriores
| Estrategia | CV RMSE | vs Anterior | Gap to Top |
|------------|---------|-------------|------------|
| Baseline | 8.78256 | - | 0.243 |
| XGBoost Tuned | 8.76100 | -0.02156 | 0.221 |
| CatBoost Default | 8.76817 | +0.00717 | 0.228 |
| Feature Engineering | 8.76936 | - | 0.229 |
| **CatBoost Tuned** | **8.74974** | **-0.01126** | **0.210** |

**Nuevo mejor modelo!** Mejora de 0.01126 sobre XGBoost tuneado.

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | study_hours | 49.37 |
| 2 | class_attendance | 21.15 |
| 3 | sleep_quality | 9.67 |
| 4 | study_method | 8.16 |
| 5 | facility_rating | 6.39 |
| 6 | sleep_hours | 4.53 |
| 7 | age | 0.24 |
| 8 | course | 0.20 |
| 9 | gender | 0.12 |
| 10 | exam_difficulty | 0.10 |
| 11 | internet_access | 0.05 |

## Observaciones

1. **Mejor modelo hasta ahora**: 8.74974 CV RMSE
2. **GPU aceleró significativamente**: ~2h 19min para 50 trials
3. **Learning rate alto**: 0.17 (mayor que XGBoost tuneado 0.06)
4. **Iteraciones moderadas**: 1660 con early stopping efectivo
5. **min_data_in_leaf=65**: Regularización fuerte para evitar overfitting
6. **class_attendance más importante**: 21% vs 9.66% en XGBoost

## Posibles Mejoras

1. Stacking con XGBoost + CatBoost
2. Target encoding adicional
3. Más trials de Optuna
4. LightGBM para ensemble

## Archivos

- `catboost_tuning.py` - Código de optimización con GPU
- `submissions/submission_catboost_tuned_20260106_002652.csv` - Submission generada
