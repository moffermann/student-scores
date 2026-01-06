#!/usr/bin/env python3
"""
Profile-Weighted Ensemble

Idea: diferentes perfiles pueden beneficiarse de diferentes pesos de ensemble.
En vez de pesos fijos (0.30/0.40/0.30), aprendemos pesos optimos por perfil.

Estrategia:
1. Generar OOF predictions de cada modelo
2. Para cada perfil (o grupo de perfiles), encontrar pesos optimos
3. Aplicar pesos aprendidos a test predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

N_FOLDS = 5

print("=" * 70)
print("PROFILE-WEIGHTED ENSEMBLE")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

target = 'exam_score'
cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
            'study_method', 'facility_rating', 'exam_difficulty']
num_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'age']

y = train[target].values
test_ids = test['id']

# Create profile
train['profile'] = train[cat_cols].astype(str).agg('_'.join, axis=1)
test['profile'] = test[cat_cols].astype(str).agg('_'.join, axis=1)

# Profile ID mapping
all_profiles = list(set(train['profile'].unique()) | set(test['profile'].unique()))
profile_to_id = {p: i for i, p in enumerate(all_profiles)}
train['profile_id'] = train['profile'].map(profile_to_id)
test['profile_id'] = test['profile'].map(profile_to_id)

n_profiles = len(all_profiles)
print(f"  Number of profiles: {n_profiles}")

# ============================================================
# PREPARE DATA FOR MODELS
# ============================================================
print("\n[2/6] Preparing data...")

# For XGBoost/HistGB: label encode
train_enc = train.copy()
test_enc = test.copy()

for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col])
    test_enc[col] = le.transform(test_enc[col])

X_train_enc = train_enc[cat_cols + num_cols].values
X_test_enc = test_enc[cat_cols + num_cols].values

# For CatBoost: original categoricals
X_train_cat = train[cat_cols + num_cols].copy()
X_test_cat = test[cat_cols + num_cols].copy()
cat_features_idx = list(range(len(cat_cols)))

# ============================================================
# MODEL PARAMS
# ============================================================

XGBOOST_PARAMS = {
    'n_estimators': 1156,
    'max_depth': 6,
    'learning_rate': 0.059577,
    'subsample': 0.801002,
    'colsample_bytree': 0.992775,
    'min_child_weight': 5,
    'gamma': 0.217669,
    'reg_alpha': 0.254105,
    'reg_lambda': 0.015326,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'
}

CATBOOST_PARAMS = {
    'iterations': 1660,
    'depth': 6,
    'learning_rate': 0.171473,
    'l2_leaf_reg': 5.500127,
    'min_data_in_leaf': 65,
    'random_strength': 2.339920,
    'bagging_temperature': 0.403898,
    'border_count': 254,
    'random_seed': 42,
    'verbose': 0,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'devices': '0'
}

HISTGB_PARAMS = {
    'max_iter': 1371,
    'max_depth': 5,
    'learning_rate': 0.066045,
    'max_leaf_nodes': 55,
    'min_samples_leaf': 20,
    'l2_regularization': 2.912878,
    'max_bins': 253,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50,
    'categorical_features': list(range(len(cat_cols)))
}

# ============================================================
# GENERATE OOF PREDICTIONS
# ============================================================
print("\n[3/6] Generating OOF predictions...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_hgb = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_hgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_enc), 1):
    X_tr_enc, X_val_enc = X_train_enc[train_idx], X_train_enc[val_idx]
    X_tr_cat, X_val_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_enc)
    test_xgb += xgb_model.predict(X_test_enc) / N_FOLDS

    # CatBoost
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_val_cat)
    test_cat += cat_model.predict(X_test_cat) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_enc, y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_val_enc)
    test_hgb += hgb_model.predict(X_test_enc) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Individual model scores
rmse_xgb = np.sqrt(np.mean((y - oof_xgb) ** 2))
rmse_cat = np.sqrt(np.mean((y - oof_cat) ** 2))
rmse_hgb = np.sqrt(np.mean((y - oof_hgb) ** 2))

print(f"\n  Individual model RMSE:")
print(f"    XGBoost:  {rmse_xgb:.5f}")
print(f"    CatBoost: {rmse_cat:.5f}")
print(f"    HistGB:   {rmse_hgb:.5f}")

# Fixed-weight ensemble
blend_fixed = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
rmse_fixed = np.sqrt(np.mean((y - blend_fixed) ** 2))
print(f"    Fixed Ensemble (0.30/0.40/0.30): {rmse_fixed:.5f}")

# ============================================================
# OPTIMIZE GLOBAL WEIGHTS
# ============================================================
print("\n[4/6] Optimizing global weights...")

def rmse_loss(weights, preds, y_true):
    w = np.abs(weights) / np.abs(weights).sum()  # Normalize
    blend = w[0] * preds[0] + w[1] * preds[1] + w[2] * preds[2]
    return np.sqrt(np.mean((y_true - blend) ** 2))

oof_preds = [oof_xgb, oof_cat, oof_hgb]

result = minimize(
    rmse_loss,
    x0=[0.33, 0.34, 0.33],
    args=(oof_preds, y),
    method='Nelder-Mead'
)

optimal_weights = np.abs(result.x) / np.abs(result.x).sum()
print(f"  Optimal global weights: XGB={optimal_weights[0]:.4f}, CAT={optimal_weights[1]:.4f}, HGB={optimal_weights[2]:.4f}")

blend_optimal = optimal_weights[0] * oof_xgb + optimal_weights[1] * oof_cat + optimal_weights[2] * oof_hgb
rmse_optimal = np.sqrt(np.mean((y - blend_optimal) ** 2))
print(f"  Optimized Ensemble RMSE: {rmse_optimal:.5f}")

# ============================================================
# PROFILE-SPECIFIC WEIGHTS
# ============================================================
print("\n[5/6] Learning profile-specific weights...")

# Stack OOF predictions
oof_stack = np.column_stack([oof_xgb, oof_cat, oof_hgb])
test_stack = np.column_stack([test_xgb, test_cat, test_hgb])

# Approach 1: Learn weights per profile using Ridge regression
# For each sample, predict y from [xgb_pred, cat_pred, hgb_pred] with profile as context

# Create profile features
profile_ids_train = train['profile_id'].values
profile_ids_test = test['profile_id'].values

# Method: Meta-model that takes (model_preds, profile) and outputs final prediction
# Use a simple stacking meta-model

# Create interaction features: pred * profile_indicator
# Too many profiles, so cluster them first

print("  Creating profile groups...")
profile_counts = train.groupby('profile_id').size()
profile_mean_target = train.groupby('profile_id')[target].mean()

# Cluster profiles by (count, mean_target)
from sklearn.cluster import KMeans

profile_features = pd.DataFrame({
    'count': profile_counts,
    'mean_target': profile_mean_target
}).fillna(0)

# Normalize
profile_features_norm = (profile_features - profile_features.mean()) / profile_features.std()

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
profile_to_cluster = kmeans.fit_predict(profile_features_norm)

# Map profiles to clusters
profile_cluster_map = dict(zip(profile_features.index, profile_to_cluster))

train['cluster'] = train['profile_id'].map(profile_cluster_map).fillna(0).astype(int)
test['cluster'] = test['profile_id'].map(profile_cluster_map).fillna(0).astype(int)

print(f"  Created {n_clusters} profile clusters")

# Learn optimal weights per cluster
cluster_weights = {}

for cluster in range(n_clusters):
    mask = train['cluster'] == cluster
    if mask.sum() < 100:
        cluster_weights[cluster] = optimal_weights
        continue

    cluster_oof = [oof_xgb[mask], oof_cat[mask], oof_hgb[mask]]
    cluster_y = y[mask]

    result = minimize(
        rmse_loss,
        x0=optimal_weights,
        args=(cluster_oof, cluster_y),
        method='Nelder-Mead'
    )

    w = np.abs(result.x) / np.abs(result.x).sum()
    cluster_weights[cluster] = w

# Apply cluster-specific weights
oof_profile = np.zeros(len(train))
test_profile = np.zeros(len(test))

for cluster in range(n_clusters):
    # Train
    mask = train['cluster'] == cluster
    w = cluster_weights[cluster]
    oof_profile[mask] = w[0] * oof_xgb[mask] + w[1] * oof_cat[mask] + w[2] * oof_hgb[mask]

    # Test
    mask_test = test['cluster'] == cluster
    test_profile[mask_test] = w[0] * test_xgb[mask_test] + w[1] * test_cat[mask_test] + w[2] * test_hgb[mask_test]

rmse_profile = np.sqrt(np.mean((y - oof_profile) ** 2))
print(f"\n  Profile-weighted Ensemble RMSE: {rmse_profile:.5f}")

# Show weight distribution across clusters
print("\n  Weights by cluster:")
for cluster in sorted(cluster_weights.keys())[:5]:
    w = cluster_weights[cluster]
    n = (train['cluster'] == cluster).sum()
    print(f"    Cluster {cluster} (n={n:,}): XGB={w[0]:.3f}, CAT={w[1]:.3f}, HGB={w[2]:.3f}")
print("    ...")

# ============================================================
# STACKING META-MODEL
# ============================================================
print("\n  Training stacking meta-model...")

# Use Ridge regression with cluster indicator as features
cluster_dummies = pd.get_dummies(train['cluster'], prefix='cluster')
meta_features = np.hstack([oof_stack, cluster_dummies.values])

test_cluster_dummies = pd.get_dummies(test['cluster'], prefix='cluster')
# Ensure same columns
for col in cluster_dummies.columns:
    if col not in test_cluster_dummies.columns:
        test_cluster_dummies[col] = 0
test_cluster_dummies = test_cluster_dummies[cluster_dummies.columns]
meta_features_test = np.hstack([test_stack, test_cluster_dummies.values])

# Train Ridge meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features, y)

oof_meta = meta_model.predict(meta_features)
test_meta = meta_model.predict(meta_features_test)

rmse_meta = np.sqrt(np.mean((y - oof_meta) ** 2))
print(f"  Stacking Meta-Model RMSE: {rmse_meta:.5f}")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 70)
print("[6/6] RESULTS")
print("=" * 70)

print("\n  Method Comparison:")
print("-" * 50)
print(f"  {'Method':<35} {'CV RMSE':>12}")
print("-" * 50)
print(f"  {'Fixed Ensemble (0.30/0.40/0.30)':<35} {rmse_fixed:>12.5f}")
print(f"  {'Optimized Global Weights':<35} {rmse_optimal:>12.5f}")
print(f"  {'Profile-Cluster Weights':<35} {rmse_profile:>12.5f}")
print(f"  {'Stacking Meta-Model':<35} {rmse_meta:>12.5f}")
print("-" * 50)
print(f"  {'Best baseline (pseudo-labeling)':<35} {'8.72348':>12}")
print("-" * 50)

# Choose best method
methods = {
    'fixed': (rmse_fixed, 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb),
    'optimal': (rmse_optimal, optimal_weights[0] * test_xgb + optimal_weights[1] * test_cat + optimal_weights[2] * test_hgb),
    'profile': (rmse_profile, test_profile),
    'meta': (rmse_meta, test_meta)
}

best_method = min(methods.items(), key=lambda x: x[1][0])
best_name, (best_rmse, best_preds) = best_method

print(f"\n  Best method: {best_name} with RMSE {best_rmse:.5f}")

# Save
predictions = np.clip(best_preds, 0, 100)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({'id': test_ids, 'exam_score': predictions})
filepath = f'{SUBMISSIONS_DIR}/submission_profile_weighted_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")
print(f"  Difference vs pseudo-labeling: {best_rmse - 8.72348:+.5f}")

print("\n" + "=" * 70)
