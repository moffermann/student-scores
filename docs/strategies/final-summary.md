# Resumen Final de Estrategias

## Mejor Modelo: PL + Meta-Stacking

**CV RMSE: 8.72238 | Public LB: 8.70545**

### Descripcion

Combina dos tecnicas:

1. **Pseudo-Labeling**: Usa predicciones del ensemble sobre test como datos de entrenamiento adicionales
2. **Meta-Stacking con Clusters de Perfil**: Un modelo Ridge que aprende pesos optimos por cluster de perfil

### Por que funciona

- Pseudo-labeling aumenta efectivamente el dataset de 630K a 900K samples
- Los clusters de perfil capturan que diferentes combinaciones categoricas se benefician de diferentes pesos de ensemble
- El meta-model aprende estas relaciones de forma automatica

## Cronologia de Experimentos

### Fase 1: Baseline y Tuning
- Baseline XGBoost: 8.78256
- Optuna tuning (50 trials): 8.76100
- CatBoost nativo: 8.76817
- CatBoost tuned GPU: 8.74974

### Fase 2: Ensembles
- XGBoost + CatBoost: 8.73771
- + HistGradientBoosting: 8.73339

### Fase 3: Pseudo-Labeling
- Primera ronda: 8.72348
- Segunda ronda: 8.72521 (sin mejora)

### Fase 4: Custom Models (exploracion)
- NN Profile-Conditioned: 8.88980
- NN con FiLM: 8.91372
- Hybrid Embeddings + Trees: 8.73686

### Fase 5: Combinacion Final
- PL + Profile Weights: 8.72348
- PL + Meta-Stacking: 8.72238 (MEJOR)

## Lecciones Aprendidas

### Lo que funciono
1. **GPU training**: CatBoost y XGBoost en GPU aceleran significativamente
2. **Ensemble de 3 modelos**: XGB + CAT + HGB se complementan bien
3. **Pseudo-labeling**: Mejora consistente de ~0.01 RMSE
4. **Meta-stacking**: Pequena mejora adicional sobre pesos fijos

### Lo que NO funciono
1. **Feature engineering manual**: Los arboles capturan interacciones automaticamente
2. **Target encoding**: CatBoost ya lo hace internamente
3. **Modelos por subgrupo**: El modelo global generaliza mejor
4. **Redes neuronales custom**: No superan a los arboles en datos tabulares
5. **Embeddings para arboles**: El manejo nativo de categoricas es superior

### Insights sobre los datos
- 5,668 perfiles unicos de 5,670 teoricos
- Rango de medias por perfil: 44-84 puntos (39 puntos de spread)
- Las correlaciones varian por perfil:
  - study_hours: 0.65-0.88
  - class_attendance: 0.08-0.62
  - sleep_hours: -0.12 a 0.43

## Progreso Total

| Metrica | Valor |
|---------|-------|
| RMSE inicial | 8.78256 |
| CV RMSE final | 8.72238 |
| **Public LB** | **8.70545** |
| Mejora total (CV) | 0.06018 |
| Mejora total (LB) | 0.07711 |
| Gap al top (~8.54) | 0.165 |
