#!/usr/bin/env python3
"""
Pseudo-Labeling + Profile-Weighted Ensemble

Combina las dos mejores estrategias:
1. Pseudo-labeling para aumentar datos de entrenamiento
2. Profile-weighted ensemble para optimizar pesos por perfil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
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
print("PSEUDO-LABELING + PROFILE-WEIGHTED ENSEMBLE")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/7] Loading data...")

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

print(f"  Profiles: {len(all_profiles)}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[2/7] Preparing data...")

# Label encode for XGBoost/HistGB
label_encoders = {}
train_enc = train.copy()
test_enc = test.copy()

for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col])
    test_enc[col] = le.transform(test_enc[col])
    label_encoders[col] = le

X_train_enc = train_enc[cat_cols + num_cols].values
X_test_enc = test_enc[cat_cols + num_cols].values

# For CatBoost
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
# STEP 1: Generate initial pseudo-labels
# ============================================================
print("\n[3/7] Generating initial pseudo-labels...")

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

# Baseline ensemble
baseline_blend = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
baseline_rmse = np.sqrt(np.mean((y - baseline_blend) ** 2))
print(f"  Baseline CV RMSE: {baseline_rmse:.5f}")

# Pseudo-labels
pseudo_labels = 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb
print(f"  Pseudo-labels generated: {len(pseudo_labels)}")

# ============================================================
# STEP 2: Train with augmented data
# ============================================================
print("\n[4/7] Training with pseudo-labeled data...")

# Augment training data
X_train_aug_enc = np.vstack([X_train_enc, X_test_enc])
X_train_aug_cat = pd.concat([X_train_cat, X_test_cat], ignore_index=True)
y_train_aug = np.concatenate([y, pseudo_labels])

# Track original vs pseudo
is_original = np.concatenate([np.ones(len(train)), np.zeros(len(test))])

print(f"  Augmented train: {len(X_train_aug_enc)} samples")
print(f"    Original: {len(train)}")
print(f"    Pseudo: {len(test)}")

# OOF predictions on original data only
oof_xgb_pl = np.zeros(len(train))
oof_cat_pl = np.zeros(len(train))
oof_hgb_pl = np.zeros(len(train))

test_xgb_pl = np.zeros(len(test))
test_cat_pl = np.zeros(len(test))
test_hgb_pl = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_enc), 1):
    # Augmented training: original train fold + ALL pseudo-labeled test
    aug_train_mask = np.zeros(len(X_train_aug_enc), dtype=bool)
    aug_train_mask[train_idx] = True  # Original train fold
    aug_train_mask[len(train):] = True  # All pseudo-labeled

    X_tr_enc = X_train_aug_enc[aug_train_mask]
    X_tr_cat = X_train_aug_cat.iloc[aug_train_mask]
    y_tr = y_train_aug[aug_train_mask]

    X_val_enc = X_train_enc[val_idx]
    X_val_cat = X_train_cat.iloc[val_idx]
    y_val = y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    oof_xgb_pl[val_idx] = xgb_model.predict(X_val_enc)
    test_xgb_pl += xgb_model.predict(X_test_enc) / N_FOLDS

    # CatBoost
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat_pl[val_idx] = cat_model.predict(X_val_cat)
    test_cat_pl += cat_model.predict(X_test_cat) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_enc, y_tr)
    oof_hgb_pl[val_idx] = hgb_model.predict(X_val_enc)
    test_hgb_pl += hgb_model.predict(X_test_enc) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb_pl[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat_pl[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb_pl[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Fixed-weight pseudo-labeling result
pl_blend = 0.30 * oof_xgb_pl + 0.40 * oof_cat_pl + 0.30 * oof_hgb_pl
pl_rmse = np.sqrt(np.mean((y - pl_blend) ** 2))
print(f"\n  Pseudo-Labeling CV RMSE (fixed weights): {pl_rmse:.5f}")

# ============================================================
# STEP 3: Profile clustering
# ============================================================
print("\n[5/7] Creating profile clusters...")

profile_counts = train.groupby('profile_id').size()
profile_mean_target = train.groupby('profile_id')[target].mean()

profile_features = pd.DataFrame({
    'count': profile_counts,
    'mean_target': profile_mean_target
}).fillna(0)

profile_features_norm = (profile_features - profile_features.mean()) / (profile_features.std() + 1e-8)

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
profile_to_cluster = kmeans.fit_predict(profile_features_norm)

profile_cluster_map = dict(zip(profile_features.index, profile_to_cluster))

train['cluster'] = train['profile_id'].map(profile_cluster_map).fillna(0).astype(int)
test['cluster'] = test['profile_id'].map(profile_cluster_map).fillna(0).astype(int)

print(f"  Created {n_clusters} profile clusters")

# ============================================================
# STEP 4: Optimize profile-weighted ensemble
# ============================================================
print("\n[6/7] Optimizing profile-specific weights...")

def rmse_loss(weights, preds, y_true):
    w = np.abs(weights) / np.abs(weights).sum()
    blend = w[0] * preds[0] + w[1] * preds[1] + w[2] * preds[2]
    return np.sqrt(np.mean((y_true - blend) ** 2))

# Global optimal weights
oof_preds_pl = [oof_xgb_pl, oof_cat_pl, oof_hgb_pl]
result = minimize(rmse_loss, x0=[0.33, 0.34, 0.33], args=(oof_preds_pl, y), method='Nelder-Mead')
optimal_weights = np.abs(result.x) / np.abs(result.x).sum()

print(f"  Global optimal weights: XGB={optimal_weights[0]:.4f}, CAT={optimal_weights[1]:.4f}, HGB={optimal_weights[2]:.4f}")

pl_optimal = optimal_weights[0] * oof_xgb_pl + optimal_weights[1] * oof_cat_pl + optimal_weights[2] * oof_hgb_pl
pl_optimal_rmse = np.sqrt(np.mean((y - pl_optimal) ** 2))
print(f"  PL + Optimal Weights RMSE: {pl_optimal_rmse:.5f}")

# Per-cluster weights
cluster_weights = {}
for cluster in range(n_clusters):
    mask = train['cluster'] == cluster
    if mask.sum() < 100:
        cluster_weights[cluster] = optimal_weights
        continue

    cluster_oof = [oof_xgb_pl[mask], oof_cat_pl[mask], oof_hgb_pl[mask]]
    cluster_y = y[mask]

    result = minimize(rmse_loss, x0=optimal_weights, args=(cluster_oof, cluster_y), method='Nelder-Mead')
    w = np.abs(result.x) / np.abs(result.x).sum()
    cluster_weights[cluster] = w

# Apply cluster-specific weights
oof_profile_pl = np.zeros(len(train))
test_profile_pl = np.zeros(len(test))

for cluster in range(n_clusters):
    w = cluster_weights[cluster]

    mask = train['cluster'] == cluster
    oof_profile_pl[mask] = w[0] * oof_xgb_pl[mask] + w[1] * oof_cat_pl[mask] + w[2] * oof_hgb_pl[mask]

    mask_test = test['cluster'] == cluster
    test_profile_pl[mask_test] = w[0] * test_xgb_pl[mask_test] + w[1] * test_cat_pl[mask_test] + w[2] * test_hgb_pl[mask_test]

pl_profile_rmse = np.sqrt(np.mean((y - oof_profile_pl) ** 2))
print(f"  PL + Profile Weights RMSE: {pl_profile_rmse:.5f}")

# Stacking meta-model
print("\n  Training stacking meta-model...")

oof_stack = np.column_stack([oof_xgb_pl, oof_cat_pl, oof_hgb_pl])
test_stack = np.column_stack([test_xgb_pl, test_cat_pl, test_hgb_pl])

cluster_dummies = pd.get_dummies(train['cluster'], prefix='cluster')
meta_features = np.hstack([oof_stack, cluster_dummies.values])

test_cluster_dummies = pd.get_dummies(test['cluster'], prefix='cluster')
for col in cluster_dummies.columns:
    if col not in test_cluster_dummies.columns:
        test_cluster_dummies[col] = 0
test_cluster_dummies = test_cluster_dummies[cluster_dummies.columns]
meta_features_test = np.hstack([test_stack, test_cluster_dummies.values])

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features, y)

oof_meta = meta_model.predict(meta_features)
test_meta = meta_model.predict(meta_features_test)

pl_meta_rmse = np.sqrt(np.mean((y - oof_meta) ** 2))
print(f"  PL + Stacking Meta-Model RMSE: {pl_meta_rmse:.5f}")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 70)
print("[7/7] RESULTS")
print("=" * 70)

print("\n  Method Comparison:")
print("-" * 55)
print(f"  {'Method':<40} {'CV RMSE':>12}")
print("-" * 55)
print(f"  {'Baseline (no PL, fixed weights)':<40} {baseline_rmse:>12.5f}")
print(f"  {'PL + Fixed Weights (0.30/0.40/0.30)':<40} {pl_rmse:>12.5f}")
print(f"  {'PL + Optimal Global Weights':<40} {pl_optimal_rmse:>12.5f}")
print(f"  {'PL + Profile-Cluster Weights':<40} {pl_profile_rmse:>12.5f}")
print(f"  {'PL + Stacking Meta-Model':<40} {pl_meta_rmse:>12.5f}")
print("-" * 55)
print(f"  {'Previous best (PL only)':<40} {'8.72348':>12}")
print("-" * 55)

# Best result
methods = {
    'pl_fixed': (pl_rmse, 0.30 * test_xgb_pl + 0.40 * test_cat_pl + 0.30 * test_hgb_pl),
    'pl_optimal': (pl_optimal_rmse, optimal_weights[0] * test_xgb_pl + optimal_weights[1] * test_cat_pl + optimal_weights[2] * test_hgb_pl),
    'pl_profile': (pl_profile_rmse, test_profile_pl),
    'pl_meta': (pl_meta_rmse, test_meta)
}

best_method = min(methods.items(), key=lambda x: x[1][0])
best_name, (best_rmse, best_preds) = best_method

print(f"\n  BEST: {best_name} with RMSE {best_rmse:.5f}")
print(f"  Improvement vs previous best: {8.72348 - best_rmse:+.5f}")

# Save
predictions = np.clip(best_preds, 0, 100)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({'id': test_ids, 'exam_score': predictions})
filepath = f'{SUBMISSIONS_DIR}/submission_pl_profile_weighted_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")
print("\n" + "=" * 70)
