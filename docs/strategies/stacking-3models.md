# Stacking Ensemble - 3 Models

**Branch:** `stacking-lightgbm`
**Fecha:** 2026-01-06
**Modelos Base:** XGBoost (GPU) + CatBoost (GPU) + HistGradientBoosting

## Descripcion

Ensemble de 3 modelos usando out-of-fold predictions. Se agrego HistGradientBoostingRegressor de sklearn como tercer modelo (alternativa a LightGBM que no estaba disponible por dependencias del sistema).

## Pipeline

1. Carga de datos
2. Preparacion separada para cada modelo
3. Tuning de HistGB con Optuna (30 trials)
4. Generacion de OOF predictions (5-fold CV) para 3 modelos
5. Prueba de metodos de blending
6. Seleccion del mejor metodo
7. Generacion de submission

## Hiperparametros

### XGBoost (del tuning previo)
```python
{
    'n_estimators': 1156,
    'max_depth': 6,
    'learning_rate': 0.059577,
    'device': 'cuda'
}
```

### CatBoost (del tuning previo)
```python
{
    'iterations': 1660,
    'depth': 6,
    'learning_rate': 0.171473,
    'task_type': 'GPU'
}
```

### HistGradientBoosting (nuevo - tuneado con Optuna)
```python
{
    'max_iter': 1371,
    'max_depth': 5,
    'learning_rate': 0.066045,
    'max_leaf_nodes': 55,
    'min_samples_leaf': 20,
    'l2_regularization': 2.912878,
    'max_bins': 253
}
```

## Resultados

### Scores Individuales
| Modelo | CV RMSE | Std |
|--------|---------|-----|
| XGBoost | 8.75590 | 0.01321 |
| CatBoost | 8.74875 | 0.01338 |
| HistGB | 8.75652 | 0.01094 |

### Metodos de Blending
| Metodo | CV RMSE |
|--------|---------|
| Simple Average (3) | 8.73352 |
| **Weighted Average (3)** | **8.73339** |
| Ridge Stacking (3) | 8.73346 |
| XGB+CAT (2) | 8.73792 |
| XGB+HGB (2) | 8.73972 |
| CAT+HGB (2) | 8.73762 |

### Pesos Optimos (3 modelos)
| Modelo | Peso Grid Search | Coef Ridge |
|--------|------------------|------------|
| XGBoost | 0.30 | 0.3024 |
| CatBoost | 0.40 | 0.3913 |
| HistGB | 0.30 | 0.3069 |

### Comparacion con Estrategias Anteriores
| Estrategia | CV RMSE | Mejora |
|------------|---------|--------|
| Baseline | 8.78256 | - |
| XGBoost Tuned | 8.76100 | -0.022 |
| CatBoost Tuned | 8.74974 | -0.011 |
| 2-Model Stacking | 8.73771 | -0.012 |
| **3-Model Stacking** | **8.73339** | **-0.004** |

**Mejora total desde baseline: 0.049 RMSE**

## Observaciones

1. **HistGB comparable a XGBoost**: 8.75652 vs 8.75590
2. **CatBoost sigue siendo el mejor individual**: 8.74875
3. **3 modelos mejor que 2**: 8.73339 vs 8.73771
4. **Pesos balanceados**: ~30/40/30 indica que los 3 aportan diversidad
5. **Ridge vs Grid Search**: Resultados practicamente identicos

## Gap Analysis

| Modelo | CV RMSE | Gap to Top |
|--------|---------|------------|
| Top Leaderboard | ~8.54 | 0.000 |
| **3-Model Stacking** | **8.73339** | **0.193** |

Aun hay ~0.19 de gap. Posibles mejoras:
- Pseudo-labeling
- Target encoding con CV
- Modelos adicionales (ExtraTrees, etc.)
- Features de diferencias entre predicciones

## Archivos

- `stacking_3models.py` - Codigo de stacking con 3 modelos
- `submissions/submission_stacking3_*.csv` - Submission generada
