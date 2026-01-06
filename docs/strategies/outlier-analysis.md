# Outlier Analysis by Course

**Branch:** `outlier-analysis`
**Fecha:** 2026-01-06
**Objetivo:** Detectar y tratar outliers de manera estratificada por curso

## Hipótesis Inicial

1. Diferentes cursos tienen diferentes patrones de "normalidad"
2. Estudiantes mayores (22-24) son más resilientes a factores como falta de sueño

## Metodología

### Métodos de Detección
1. **MAD (Median Absolute Deviation)**: `|x - median| / (1.4826 * MAD) > 3`
2. **IQR (Interquartile Range)**: `fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

### Features Analizadas
- `study_hours` (0.1 - 7.9)
- `class_attendance` (40.6 - 99.4)
- `sleep_hours` (4.1 - 9.9)
- `age` (17 - 24)

## Resultados

### Hallazgo Principal: NO HAY OUTLIERS

| Método | Outliers Detectados |
|--------|---------------------|
| MAD (threshold=3) | 0 |
| IQR (k=1.5) | 0 |

Los datos ya están completamente limpios y acotados. Cada feature numérica tiene rangos bien definidos y no hay valores extremos.

### Distribuciones por Curso

Todas las distribuciones son prácticamente **idénticas entre cursos**:

| Feature | Mean (all courses) | Std (all courses) | Range |
|---------|-------------------|-------------------|-------|
| study_hours | 3.92 - 4.05 | 2.33 - 2.40 | [0.1, 7.9] |
| class_attendance | 71.39 - 72.43 | 17.27 - 17.66 | [40.6, 99.4] |
| sleep_hours | 7.02 - 7.12 | 1.73 - 1.76 | [4.1, 9.9] |
| age | 20.51 - 20.56 | 2.23 - 2.27 | [17, 24] |

### Análisis de Resiliencia por Edad

#### Exam Score por Grupo de Edad y Calidad de Sueño
| Age Group | Poor Sleep | Average Sleep | Good Sleep |
|-----------|------------|---------------|------------|
| 17-18 | 56.52 | 62.59 | 67.92 |
| 19-21 | 57.02 | 62.50 | 67.68 |
| 22-24 | 57.28 | 62.87 | 68.07 |

#### Impacto de Pocas Horas de Sueño (<6h)
| Age Group | Con bajo sueño | Normal | Impacto |
|-----------|----------------|--------|---------|
| 17-18 | 58.50 | 64.03 | **-5.53** |
| 19-21 | 58.70 | 64.10 | **-5.41** |
| 22-24 | 59.04 | 64.47 | **-5.43** |

**Conclusión**: La hipótesis de resiliencia por edad **no se confirma significativamente**. El impacto de la falta de sueño es similar en todos los grupos (~5.4-5.5 puntos).

## Visualizaciones Generadas

- `analysis/boxplots_by_course.png` - Distribuciones por curso
- `analysis/age_sleep_heatmap.png` - Heatmap edad vs calidad de sueño
- `analysis/outliers_by_course.png` - Conteo de outliers (todos en 0)

## Conclusiones

1. **Los datos están limpios**: No hay outliers que tratar
2. **Cursos homogéneos**: Las distribuciones son casi idénticas entre cursos
3. **Resiliencia por edad**: Efecto mínimo (~0.1 puntos diferencia)
4. **Estrategia de capping**: **NO APLICABLE** - no hay valores extremos

## Implicaciones para el Modelo

Dado que no hay outliers ni diferencias significativas por curso:
- El tratamiento de outliers **no mejorará** el modelo
- Las features de interacción curso-específicas probablemente no ayuden
- Debemos buscar mejoras en otras direcciones

## Siguiente Paso Recomendado

Ya que la calidad de datos no es el problema, explorar:
1. **Ensemble/Stacking**: Combinar XGBoost + CatBoost
2. **Pseudo-labeling**: Usar predicciones del test como datos adicionales
3. **Diferentes encodings**: Target encoding con cross-validation

## Archivos

- `outlier_analysis.py` - Script de análisis
- `analysis/*.png` - Visualizaciones generadas
