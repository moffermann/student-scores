# CatBoost Native Categorical Strategy

**Branch:** `catboost-native`
**Fecha:** 2026-01-05
**Modelo:** CatBoost Regressor con categóricas nativas

## Descripción

CatBoost con manejo nativo de variables categóricas (sin encoding manual). CatBoost usa internamente target statistics con regularización para evitar overfitting.

## Pipeline

1. Carga de datos
2. Identificación de columnas categóricas (mantener como strings)
3. Crear Pool con cat_features
4. 5-Fold CV con early stopping
5. Entrenamiento final
6. Predicciones con clipping

## Features

Sin transformación manual:
- 7 features categóricas (nativas)
- 4 features numéricas

**Total features:** 11

## Hiperparámetros

```python
catboost_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.1,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 50
}
```

## Resultados

### Cross Validation (5-Fold)
| Fold | RMSE | Best Iteration |
|------|------|----------------|
| 1 | 8.75781 | 999 |
| 2 | 8.76640 | 999 |
| 3 | 8.75300 | 999 |
| 4 | 8.77205 | 999 |
| 5 | 8.79160 | 998 |
| **Mean** | **8.76817** | 999 |
| **Std** | 0.01345 | - |

### Comparación con Estrategias Anteriores
| Estrategia | CV RMSE | vs Tuned XGBoost |
|------------|---------|------------------|
| Baseline | 8.78256 | +0.02156 |
| Hyperparameter Tuning (XGBoost) | 8.76100 | - |
| Feature Engineering | 8.76936 | +0.00836 |
| **CatBoost Native** | **8.76817** | **+0.00717** |

**Resultado:** Peor que XGBoost tuneado por 0.00717

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | study_hours | 49.44 |
| 2 | class_attendance | 20.61 |
| 3 | sleep_quality | 10.15 |
| 4 | study_method | 8.43 |
| 5 | facility_rating | 6.65 |
| 6 | sleep_hours | 4.42 |
| 7 | course | 0.12 |
| 8 | age | 0.11 |
| 9 | exam_difficulty | 0.03 |
| 10 | gender | 0.02 |
| 11 | internet_access | 0.01 |

### Diferencias vs XGBoost
| Feature | XGBoost | CatBoost | Diferencia |
|---------|---------|----------|------------|
| study_hours | 57.07% | 49.44% | -7.63% |
| class_attendance | 9.66% | 20.61% | +10.95% |
| sleep_quality | 11.35% | 10.15% | -1.20% |

CatBoost da más importancia a `class_attendance`.

## Observaciones

1. **No mejoró vs XGBoost tuneado**: 8.76817 vs 8.76100
2. **Early stopping no actuó**: Todos los folds llegaron a iter 999
3. **Necesita más iteraciones**: El modelo no convergió completamente
4. **Podría beneficiarse de tuning**: Los parámetros son defaults
5. **class_attendance más importante**: CatBoost la valora más que XGBoost

## Posibles Mejoras

1. Aumentar `iterations` a 2000+
2. Optimizar hiperparámetros con Optuna
3. Probar `depth` más alto (8-10)
4. Ajustar `learning_rate` más bajo con más iteraciones

## Archivos

- `catboost_native.py` - Código del modelo
- `submissions/submission_catboost_20260105_214915.csv` - Submission generada
