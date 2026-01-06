#!/usr/bin/env python3
"""
Experiment: Train model separately for Male vs Female
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'

print("=" * 70)
print("EXPERIMENT: Male vs Female vs Global Model")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/4] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

# Check gender distribution
print(f"\n  Gender distribution:")
for g in train['gender'].unique():
    n = len(train[train['gender'] == g])
    print(f"    {g}: {n:,} ({n/len(train)*100:.1f}%)")

# ============================================================
# BEST PARAMETERS
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
    'n_iter_no_change': 50
}

# ============================================================
# HELPER FUNCTION
# ============================================================

def train_ensemble_cv(X_train_df, y, n_folds=5):
    """Train 3-model ensemble and return CV RMSE"""

    X_enc = X_train_df.copy()
    cat_cols = X_enc.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col])

    X_xgb = X_enc.values
    X_hgb = X_enc.copy()
    cat_features_hgb = [X_hgb.columns.get_loc(col) for col in cat_cols]
    HISTGB_PARAMS['categorical_features'] = cat_features_hgb

    X_cat = X_train_df.copy()
    cat_features_idx = [X_cat.columns.get_loc(col) for col in cat_cols]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(X_train_df))
    oof_cat = np.zeros(len(X_train_df))
    oof_hgb = np.zeros(len(X_train_df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb), 1):
        X_tr_xgb, X_val_xgb = X_xgb[train_idx], X_xgb[val_idx]
        X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_tr_hgb, X_val_hgb = X_hgb.iloc[train_idx], X_hgb.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val_xgb)

        # CatBoost
        train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_idx)
        val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
        cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
        cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof_cat[val_idx] = cat_model.predict(X_val_cat)

        # HistGB
        hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
        hgb_model.fit(X_tr_hgb, y_tr)
        oof_hgb[val_idx] = hgb_model.predict(X_val_hgb)

    rmse_xgb = np.sqrt(np.mean((y - oof_xgb) ** 2))
    rmse_cat = np.sqrt(np.mean((y - oof_cat) ** 2))
    rmse_hgb = np.sqrt(np.mean((y - oof_hgb) ** 2))

    blend = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
    rmse_ensemble = np.sqrt(np.mean((y - blend) ** 2))

    return {
        'xgb': rmse_xgb,
        'cat': rmse_cat,
        'hgb': rmse_hgb,
        'ensemble': rmse_ensemble
    }

# ============================================================
# EXPERIMENT 1: Global model
# ============================================================
print("\n[2/4] Training global model (all genders)...")

target = 'exam_score'
X_all = train.drop(['id', target], axis=1)
y_all = train[target].values

print(f"  Samples: {len(X_all):,}")

global_results = train_ensemble_cv(X_all, y_all)

print(f"\n  Global Model Results:")
print(f"    XGBoost:  {global_results['xgb']:.5f}")
print(f"    CatBoost: {global_results['cat']:.5f}")
print(f"    HistGB:   {global_results['hgb']:.5f}")
print(f"    Ensemble: {global_results['ensemble']:.5f}")

# ============================================================
# EXPERIMENT 2: Per-gender models
# ============================================================
print("\n[3/4] Training per-gender models...")

genders = sorted(train['gender'].unique())
gender_results = {}

for gender in genders:
    print(f"\n  === {gender.upper()} ===")

    gender_mask = train['gender'] == gender
    X_gender = train[gender_mask].drop(['id', target], axis=1)
    y_gender = train[gender_mask][target].values

    print(f"    Samples: {len(X_gender):,} ({len(X_gender)/len(train)*100:.1f}%)")

    # Remove gender column since it's constant
    X_gender_no_gender = X_gender.drop('gender', axis=1)

    results = train_ensemble_cv(X_gender_no_gender, y_gender)
    gender_results[gender] = results

    print(f"    XGBoost:  {results['xgb']:.5f}")
    print(f"    CatBoost: {results['cat']:.5f}")
    print(f"    HistGB:   {results['hgb']:.5f}")
    print(f"    Ensemble: {results['ensemble']:.5f}")

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("[4/4] ANALYSIS")
print("=" * 70)

print("\n  COMPARISON TABLE:")
print("-" * 70)
print(f"  {'Gender':<12} | {'Samples':>10} | {'XGB':>8} | {'CAT':>8} | {'HGB':>8} | {'Ensemble':>10}")
print("-" * 70)
print(f"  {'GLOBAL':<12} | {len(train):>10,} | {global_results['xgb']:>8.5f} | "
      f"{global_results['cat']:>8.5f} | {global_results['hgb']:>8.5f} | {global_results['ensemble']:>10.5f}")
print("-" * 70)

for gender in genders:
    n_samples = len(train[train['gender'] == gender])
    r = gender_results[gender]
    diff = r['ensemble'] - global_results['ensemble']
    print(f"  {gender:<12} | {n_samples:>10,} | {r['xgb']:>8.5f} | "
          f"{r['cat']:>8.5f} | {r['hgb']:>8.5f} | {r['ensemble']:>10.5f} ({diff:+.5f})")

print("-" * 70)

# Calculate weighted average
print("\n  WEIGHTED AVERAGE OF PER-GENDER MODELS:")
total_samples = len(train)
weighted_ensemble = 0
for gender in genders:
    n_samples = len(train[train['gender'] == gender])
    weight = n_samples / total_samples
    weighted_ensemble += weight * gender_results[gender]['ensemble']

print(f"    Weighted Avg RMSE: {weighted_ensemble:.5f}")
print(f"    Global Model RMSE: {global_results['ensemble']:.5f}")
print(f"    Difference:        {weighted_ensemble - global_results['ensemble']:+.5f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if weighted_ensemble < global_results['ensemble']:
    print(f"\n  Per-gender models are BETTER by {global_results['ensemble'] - weighted_ensemble:.5f}")
    print("  Consider training separate models per gender.")
else:
    print(f"\n  Global model is BETTER by {weighted_ensemble - global_results['ensemble']:.5f}")
    print("  The global model generalizes well across genders.")

print("\n" + "=" * 70)
