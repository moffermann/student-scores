#!/usr/bin/env python3
"""
Explorar el espacio de perfiles categoricos
"""

import pandas as pd
import numpy as np

DATA_DIR = '/home/mauro/kaggle/student-scores/data'

print("=" * 70)
print("EXPLORACION DE PERFILES CATEGORICOS")
print("=" * 70)

# Load data
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"\nTrain: {len(train):,} samples")
print(f"Test:  {len(test):,} samples")

cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
            'study_method', 'facility_rating', 'exam_difficulty']
num_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'age']

# Cardinalidad de cada columna categorica
print("\n" + "=" * 70)
print("CARDINALIDAD DE VARIABLES CATEGORICAS")
print("=" * 70)

total_combinations = 1
for col in cat_cols:
    unique_vals = train[col].unique()
    n_unique = len(unique_vals)
    total_combinations *= n_unique
    print(f"\n{col}: {n_unique} valores")
    for val in sorted(unique_vals):
        count = len(train[train[col] == val])
        pct = count / len(train) * 100
        print(f"  {val}: {count:,} ({pct:.1f}%)")

print(f"\n{'=' * 70}")
print(f"COMBINACIONES TEORICAS MAXIMAS: {total_combinations:,}")
print("=" * 70)

# Crear perfil combinado
train['profile'] = train[cat_cols].astype(str).agg('_'.join, axis=1)
test['profile'] = test[cat_cols].astype(str).agg('_'.join, axis=1)

n_profiles_train = train['profile'].nunique()
n_profiles_test = test['profile'].nunique()

print(f"\nPerfiles unicos en train: {n_profiles_train:,}")
print(f"Perfiles unicos en test:  {n_profiles_test:,}")

# Perfiles en test que no estan en train
train_profiles = set(train['profile'].unique())
test_profiles = set(test['profile'].unique())
unseen_profiles = test_profiles - train_profiles

print(f"Perfiles en test NO vistos en train: {len(unseen_profiles):,}")

# Distribucion de samples por perfil
profile_counts = train['profile'].value_counts()

print(f"\n{'=' * 70}")
print("DISTRIBUCION DE SAMPLES POR PERFIL")
print("=" * 70)

print(f"\nEstadisticas:")
print(f"  Min samples:    {profile_counts.min()}")
print(f"  Max samples:    {profile_counts.max()}")
print(f"  Mean samples:   {profile_counts.mean():.1f}")
print(f"  Median samples: {profile_counts.median():.1f}")
print(f"  Std samples:    {profile_counts.std():.1f}")

# Distribucion por rangos
ranges = [(1, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, 5000), (5001, float('inf'))]
print(f"\nPerfiles por rango de samples:")
for low, high in ranges:
    count = ((profile_counts >= low) & (profile_counts <= high)).sum()
    pct = count / len(profile_counts) * 100
    if high == float('inf'):
        print(f"  {low}+: {count} perfiles ({pct:.1f}%)")
    else:
        print(f"  {low}-{high}: {count} perfiles ({pct:.1f}%)")

# Analisis de target por perfil
print(f"\n{'=' * 70}")
print("VARIABILIDAD DEL TARGET POR PERFIL")
print("=" * 70)

profile_stats = train.groupby('profile')['exam_score'].agg(['mean', 'std', 'count'])
profile_stats = profile_stats[profile_stats['count'] >= 50]  # Solo perfiles con 50+ samples

print(f"\nPerfiles con 50+ samples: {len(profile_stats):,}")
print(f"\nVariabilidad del target:")
print(f"  Rango de medias: {profile_stats['mean'].min():.2f} - {profile_stats['mean'].max():.2f}")
print(f"  Diferencia max:  {profile_stats['mean'].max() - profile_stats['mean'].min():.2f}")
print(f"  Std promedio dentro de perfiles: {profile_stats['std'].mean():.2f}")

# Top 5 perfiles con mayor y menor media
print(f"\nTop 5 perfiles con MAYOR media de exam_score:")
top5 = profile_stats.nlargest(5, 'mean')
for profile, row in top5.iterrows():
    print(f"  {row['mean']:.2f} (n={int(row['count']):,}): {profile[:60]}...")

print(f"\nTop 5 perfiles con MENOR media de exam_score:")
bottom5 = profile_stats.nsmallest(5, 'mean')
for profile, row in bottom5.iterrows():
    print(f"  {row['mean']:.2f} (n={int(row['count']):,}): {profile[:60]}...")

# Correlacion de numericas con target, por perfil
print(f"\n{'=' * 70}")
print("CORRELACION NUMERICAS-TARGET (global vs por perfil)")
print("=" * 70)

print("\nCorrelacion GLOBAL:")
for col in num_cols:
    corr = train[col].corr(train['exam_score'])
    print(f"  {col}: {corr:.4f}")

# Calcular correlacion por perfil y ver variabilidad
print("\nVariabilidad de correlaciones POR PERFIL (perfiles con n>=100):")
large_profiles = profile_stats[profile_stats['count'] >= 100].index.tolist()

for col in num_cols:
    corrs = []
    for profile in large_profiles:
        mask = train['profile'] == profile
        if mask.sum() >= 100:
            corr = train.loc[mask, col].corr(train.loc[mask, 'exam_score'])
            if not np.isnan(corr):
                corrs.append(corr)

    if corrs:
        print(f"  {col}:")
        print(f"    Min: {min(corrs):.4f}, Max: {max(corrs):.4f}, Std: {np.std(corrs):.4f}")

print("\n" + "=" * 70)
