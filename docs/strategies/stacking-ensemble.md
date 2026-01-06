# Stacking Ensemble Strategy

**Branch:** `stacking-ensemble`
**Fecha:** 2026-01-06
**Modelos Base:** XGBoost (GPU) + CatBoost (GPU)

## Descripcion

Ensemble de XGBoost y CatBoost usando out-of-fold predictions para evitar leakage. Se probaron tres metodos de combinacion.

## Pipeline

1. Carga de datos
2. Preparacion separada para cada modelo:
   - XGBoost: Label Encoding
   - CatBoost: Categoricas nativas
3. Generacion de OOF predictions (5-fold CV)
4. Prueba de metodos de blending
5. Seleccion del mejor metodo
6. Generacion de submission

## Hiperparametros Base

### XGBoost (del hyperparameter tuning)
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

### CatBoost (del CatBoost tuning)
```python
{
    'iterations': 1660,
    'depth': 6,
    'learning_rate': 0.171473,
    'l2_leaf_reg': 5.500127,
    'min_data_in_leaf': 65,
    'random_strength': 2.339920,
    'bagging_temperature': 0.403898,
    'border_count': 254,
    'task_type': 'GPU'
}
```

## Resultados

### Scores Individuales por Fold
| Fold | XGBoost | CatBoost |
|------|---------|----------|
| 1 | 8.74260 | 8.73657 |
| 2 | 8.75030 | 8.74466 |
| 3 | 8.74424 | 8.73526 |
| 4 | 8.76563 | 8.75510 |
| 5 | 8.77672 | 8.77015 |
| **Mean** | **8.75590** | **8.74835** |

### Metodos de Blending
| Metodo | CV RMSE | Descripcion |
|--------|---------|-------------|
| Simple Average | 8.73796 | (XGB + CAT) / 2 |
| **Weighted Average** | **8.73771** | 0.43*XGB + 0.57*CAT |
| Ridge Stacking | 8.73776 | Meta-learner Ridge |

### Comparacion con Estrategias Anteriores
| Estrategia | CV RMSE | vs Anterior | Gap to Top |
|------------|---------|-------------|------------|
| Baseline | 8.78256 | - | 0.243 |
| XGBoost Tuned | 8.76100 | -0.02156 | 0.221 |
| CatBoost Tuned | 8.74974 | -0.01126 | 0.210 |
| **Stacking Ensemble** | **8.73771** | **-0.01203** | **0.198** |

**Nuevo mejor modelo!** Mejora de 0.01203 sobre CatBoost tuneado.

## Pesos Optimos

| Modelo | Peso Optimo | Ridge Coef |
|--------|-------------|------------|
| XGBoost | 0.43 | 0.4335 |
| CatBoost | 0.57 | 0.5670 |

Los pesos de Ridge son practicamente identicos a los optimizados por grid search, validando la consistencia.

## Observaciones

1. **CatBoost ligeramente mejor**: 8.74835 vs 8.75590 (XGBoost)
2. **Ensemble siempre mejor**: Todos los metodos superan a los modelos individuales
3. **Diferencias minimas**: Los tres metodos de blending dan resultados muy similares (~0.0002)
4. **Diversidad de modelos**: La mejora viene de la diversidad en las predicciones

## Posibles Mejoras

1. Agregar LightGBM como tercer modelo base
2. Usar mas folds (10-fold CV)
3. Agregar features de segundo nivel (diferencias entre predicciones)
4. Neural Network como meta-learner

## Archivos

- `stacking_ensemble.py` - Codigo de stacking
- `submissions/submission_stacking_*.csv` - Submission del ensemble
- `submissions/submission_xgb_oof_*.csv` - Predicciones OOF de XGBoost
- `submissions/submission_cat_oof_*.csv` - Predicciones OOF de CatBoost
