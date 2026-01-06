#!/usr/bin/env python3
"""
Target Encoding with proper K-Fold CV to avoid leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("TARGET ENCODING WITH CV")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

target = 'exam_score'
y = train[target].values
test_ids = test['id']

cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
            'study_method', 'facility_rating', 'exam_difficulty']
num_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'age']

print(f"  Categorical columns: {len(cat_cols)}")
print(f"  Numerical columns: {len(num_cols)}")

# ============================================================
# TARGET ENCODING FUNCTION
# ============================================================

def target_encode_cv(train_df, test_df, cat_cols, target_col, n_splits=5, smoothing=10):
    """
    Target encoding with K-Fold CV to prevent leakage.

    For train: use out-of-fold target means
    For test: use full train target means

    Smoothing formula: (count * mean + global_mean * smoothing) / (count + smoothing)
    """

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    global_mean = train_df[target_col].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        print(f"    Encoding {col}...")

        # Initialize train column with global mean
        train_encoded[f'{col}_te'] = global_mean

        # Out-of-fold encoding for train
        for train_idx, val_idx in kf.split(train_df):
            # Calculate means from training fold only
            train_fold = train_df.iloc[train_idx]

            # Group stats
            agg = train_fold.groupby(col)[target_col].agg(['mean', 'count'])

            # Apply smoothing
            smoothed_mean = (agg['count'] * agg['mean'] + global_mean * smoothing) / (agg['count'] + smoothing)

            # Map to validation fold
            train_encoded.loc[train_encoded.index[val_idx], f'{col}_te'] = \
                train_df.iloc[val_idx][col].map(smoothed_mean).fillna(global_mean)

        # For test: use full train data
        agg = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed_mean = (agg['count'] * agg['mean'] + global_mean * smoothing) / (agg['count'] + smoothing)
        test_encoded[f'{col}_te'] = test_df[col].map(smoothed_mean).fillna(global_mean)

    return train_encoded, test_encoded

# ============================================================
# APPLY TARGET ENCODING
# ============================================================
print("\n[2/6] Applying target encoding...")

train_te, test_te = target_encode_cv(train, test, cat_cols, target, n_splits=5, smoothing=10)

# New feature columns
te_cols = [f'{col}_te' for col in cat_cols]
print(f"\n  Created {len(te_cols)} target encoded features")

# Show sample of encoded values
print("\n  Sample target encoded values (course):")
sample = train_te.groupby('course')['course_te'].first()
for course, val in sample.items():
    print(f"    {course}: {val:.4f}")

# ============================================================
# PREPARE FEATURES
# ============================================================
print("\n[3/6] Preparing features...")

# Option 1: Replace categoricals with target encoding
feature_cols_te_only = num_cols + te_cols

# Option 2: Keep both (label encoded + target encoded)
train_combined = train_te.copy()
test_combined = test_te.copy()

for col in cat_cols:
    le = LabelEncoder()
    train_combined[f'{col}_le'] = le.fit_transform(train_combined[col])
    test_combined[f'{col}_le'] = le.transform(test_combined[col])

le_cols = [f'{col}_le' for col in cat_cols]
feature_cols_combined = num_cols + te_cols + le_cols

print(f"  Features (TE only): {len(feature_cols_te_only)}")
print(f"  Features (Combined): {len(feature_cols_combined)}")

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
# EXPERIMENT 1: Target Encoding Only
# ============================================================
print("\n[4/6] Experiment 1: Target Encoding Only (no label encoding)...")

X_train_te = train_te[feature_cols_te_only].values
X_test_te = test_te[feature_cols_te_only].values

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_hgb = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_hgb = np.zeros(len(test))

print("\n  Training 3-model ensemble...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_te), 1):
    X_tr, X_val = X_train_te[train_idx], X_train_te[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test_te) / N_FOLDS

    # CatBoost (no cat features - all numeric now)
    cat_params = CATBOOST_PARAMS.copy()
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat += cat_model.predict(X_test_te) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_val)
    test_hgb += hgb_model.predict(X_test_te) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Calculate scores
rmse_xgb_te = np.sqrt(np.mean((y - oof_xgb) ** 2))
rmse_cat_te = np.sqrt(np.mean((y - oof_cat) ** 2))
rmse_hgb_te = np.sqrt(np.mean((y - oof_hgb) ** 2))

blend_te = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
rmse_ensemble_te = np.sqrt(np.mean((y - blend_te) ** 2))

print(f"\n  Results (TE Only):")
print(f"    XGBoost:  {rmse_xgb_te:.5f}")
print(f"    CatBoost: {rmse_cat_te:.5f}")
print(f"    HistGB:   {rmse_hgb_te:.5f}")
print(f"    Ensemble: {rmse_ensemble_te:.5f}")

# ============================================================
# EXPERIMENT 2: Combined (TE + Label Encoding)
# ============================================================
print("\n[5/6] Experiment 2: Combined (TE + Label Encoding)...")

X_train_comb = train_combined[feature_cols_combined].values
X_test_comb = test_combined[feature_cols_combined].values

oof_xgb2 = np.zeros(len(train))
oof_cat2 = np.zeros(len(train))
oof_hgb2 = np.zeros(len(train))

test_xgb2 = np.zeros(len(test))
test_cat2 = np.zeros(len(test))
test_hgb2 = np.zeros(len(test))

print("\n  Training 3-model ensemble...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_comb), 1):
    X_tr, X_val = X_train_comb[train_idx], X_train_comb[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb2[val_idx] = xgb_model.predict(X_val)
    test_xgb2 += xgb_model.predict(X_test_comb) / N_FOLDS

    # CatBoost
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat2[val_idx] = cat_model.predict(X_val)
    test_cat2 += cat_model.predict(X_test_comb) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr, y_tr)
    oof_hgb2[val_idx] = hgb_model.predict(X_val)
    test_hgb2 += hgb_model.predict(X_test_comb) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb2[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat2[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb2[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Calculate scores
rmse_xgb_comb = np.sqrt(np.mean((y - oof_xgb2) ** 2))
rmse_cat_comb = np.sqrt(np.mean((y - oof_cat2) ** 2))
rmse_hgb_comb = np.sqrt(np.mean((y - oof_hgb2) ** 2))

blend_comb = 0.30 * oof_xgb2 + 0.40 * oof_cat2 + 0.30 * oof_hgb2
rmse_ensemble_comb = np.sqrt(np.mean((y - blend_comb) ** 2))

print(f"\n  Results (Combined):")
print(f"    XGBoost:  {rmse_xgb_comb:.5f}")
print(f"    CatBoost: {rmse_cat_comb:.5f}")
print(f"    HistGB:   {rmse_hgb_comb:.5f}")
print(f"    Ensemble: {rmse_ensemble_comb:.5f}")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("[6/6] COMPARISON")
print("=" * 60)

print("\n  Method Comparison:")
print("-" * 50)
print(f"  {'Method':<25} {'Ensemble RMSE':>15}")
print("-" * 50)
print(f"  {'Baseline (no TE)':<25} {'8.73339':>15}")
print(f"  {'Target Encoding Only':<25} {rmse_ensemble_te:>15.5f}")
print(f"  {'TE + Label Encoding':<25} {rmse_ensemble_comb:>15.5f}")
print("-" * 50)

best_te = min(rmse_ensemble_te, rmse_ensemble_comb)
improvement = 8.73339 - best_te

if improvement > 0:
    print(f"\n  IMPROVEMENT: {improvement:.5f}")
else:
    print(f"\n  NO IMPROVEMENT: {improvement:.5f}")

# Select best and save
if rmse_ensemble_te < rmse_ensemble_comb:
    best_pred = 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb
    best_method = "TE Only"
else:
    best_pred = 0.30 * test_xgb2 + 0.40 * test_cat2 + 0.30 * test_hgb2
    best_method = "TE + LE"

predictions = np.clip(best_pred, 0, 100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_target_encoding_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved ({best_method}): {filepath}")

print("\n" + "=" * 60)
