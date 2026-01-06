# Feature Engineering Strategy

**Branch:** `feature-engineering`
**Fecha:** 2026-01-05
**Modelo:** XGBoost Regressor con features adicionales

## Descripción

Creación de features adicionales mediante interacciones, ratios, transformaciones polinomiales y binning. Se utilizan los hiperparámetros optimizados de la estrategia anterior.

## Pipeline

1. Carga de datos
2. Creación de 15 nuevas features
3. Label Encoding para variables categóricas
4. Entrenamiento con hiperparámetros optimizados
5. Clipping de predicciones
6. Generación de submission

## Nuevas Features Creadas

### Interacciones (3)
| Feature | Fórmula |
|---------|---------|
| study_x_attendance | study_hours × class_attendance |
| study_x_sleep_hours | study_hours × sleep_hours |
| study_per_age | study_hours / age |

### Ratios (2)
| Feature | Fórmula |
|---------|---------|
| attendance_per_study | class_attendance / (study_hours + 1) |
| sleep_study_ratio | sleep_hours / (study_hours + 1) |

### Polinomiales (2)
| Feature | Fórmula |
|---------|---------|
| study_hours_sq | study_hours² |
| attendance_sq | class_attendance² |

### Agregaciones (2)
| Feature | Fórmula |
|---------|---------|
| total_effort | study_hours + class_attendance |
| sleep_efficiency | study_hours / (sleep_hours + 1) |

### Binning (2)
| Feature | Descripción |
|---------|-------------|
| study_hours_bin | study_hours en 5 bins |
| age_group | age en 4 grupos |

### Booleanas (4)
| Feature | Condición |
|---------|-----------|
| high_study | study_hours > mediana |
| high_attendance | class_attendance > mediana |
| good_sleep | sleep_hours >= 7 |
| dedicated_student | high_study AND high_attendance |

**Total features:** 26 (11 originales + 15 nuevas)

## Hiperparámetros

Mismos que hyperparameter-tuning:

```python
xgb_params = {
    'n_estimators': 485,
    'max_depth': 7,
    'learning_rate': 0.061553,
    'subsample': 0.897410,
    'colsample_bytree': 0.817250,
    'min_child_weight': 10,
    'reg_alpha': 0.003965,
    'reg_lambda': 0.000068,
    'gamma': 0.258847
}
```

## Resultados

### Cross Validation (5-Fold)
| Fold | RMSE |
|------|------|
| 1 | 8.75876 |
| 2 | 8.76357 |
| 3 | 8.75872 |
| 4 | 8.77526 |
| 5 | 8.79050 |
| **Mean** | **8.76936** |
| **Std** | 0.01217 |

### Comparación con Estrategias Anteriores
| Estrategia | CV RMSE | Diferencia |
|------------|---------|------------|
| Baseline | 8.78256 | - |
| Hyperparameter Tuning | 8.76100 | -0.02156 |
| **Feature Engineering** | **8.76936** | **+0.00836** |

**Resultado:** Peor que hyperparameter tuning por 0.00836

## Feature Importance (Top 15)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | study_x_attendance | 0.6120 |
| 2 | sleep_quality | 0.0653 |
| 3 | study_hours | 0.0519 |
| 4 | study_method | 0.0469 |
| 5 | good_sleep | 0.0410 |
| 6 | study_x_sleep_hours | 0.0370 |
| 7 | facility_rating | 0.0368 |
| 8 | high_attendance | 0.0290 |
| 9 | total_effort | 0.0233 |
| 10 | study_hours_bin | 0.0191 |
| 11 | sleep_hours | 0.0080 |
| 12 | study_hours_sq | 0.0060 |
| 13 | class_attendance | 0.0055 |
| 14 | attendance_sq | 0.0033 |
| 15 | high_study | 0.0028 |

## Observaciones

1. **No mejora el score**: Feature engineering empeoró ligeramente el RMSE
2. **study_x_attendance domina**: La interacción captura 61.2% de importancia
3. **XGBoost ya captura interacciones**: El modelo tree-based puede aprender estas relaciones automáticamente
4. **Posible overfitting**: Más features pueden añadir ruido sin beneficio
5. **Hiperparámetros no reoptimizados**: Los params fueron optimizados para 11 features, no 26

## Lecciones Aprendidas

- Para modelos tree-based, feature engineering manual puede ser redundante
- Las interacciones ya son capturadas implícitamente por los árboles
- Mejor enfoque: probar target encoding o embeddings para categóricas

## Archivos

- `feature_engineering.py` - Código del modelo
- `submissions/submission_features_20260105_211522.csv` - Submission generada
