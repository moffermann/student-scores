# Pseudo-Labeling Strategy

## Objetivo
Aumentar el tamaño del dataset de entrenamiento usando predicciones del modelo sobre datos de test como pseudo-etiquetas.

## Metodologia

### Round 1: Generacion de Pseudo-Labels
1. Entrenar ensemble (XGB + CatBoost + HistGB) con 5-fold CV
2. Predecir sobre test set
3. Usar predicciones como pseudo-labels

### Round 2: Entrenamiento Aumentado
1. Combinar train original (630K) + test pseudo-labeled (270K) = 900K samples
2. Entrenar nuevo ensemble
3. **Importante**: En CV, solo validar sobre datos originales (evitar leakage)

### Round 3: Segunda Iteracion (opcional)
1. Generar nuevas pseudo-labels con modelo mejorado
2. Re-entrenar con datos actualizados

## Resultados

| Ronda | CV RMSE | Mejora |
|-------|---------|--------|
| Baseline | 8.73316 | - |
| **Round 1** | **8.72348** | **-0.00969** |
| Round 2 | 8.72521 | -0.00795 |

## Hallazgos

1. **Primera ronda es la mejor**: La segunda iteracion no mejora mas
2. **Mejora modesta pero consistente**: ~0.01 RMSE
3. **Sin riesgo de leakage**: La validacion se hace solo sobre datos originales

## Consideraciones Tecnicas

### Prevencion de Leakage en CV
```python
# Solo excluir datos originales del fold de validacion
aug_train_mask = np.ones(len(X_train_aug), dtype=bool)
aug_train_mask[val_idx] = False  # Excluir validacion original
# Los pseudo-labels del test siempre estan en entrenamiento
```

### Pesos del Ensemble
- XGBoost: 30%
- CatBoost: 40%
- HistGB: 30%

## Archivo
- `pseudo_labeling.py`

## Submission
- `submissions/submission_pseudo_labeling_*.csv`

## Conclusiones

Pseudo-labeling funciona bien para este problema. La mejora de 0.01 RMSE nos acerca al objetivo del top del leaderboard.

**Nuevo mejor CV RMSE: 8.72348**
