#!/usr/bin/env python3
"""
Pseudo-Labeling: Use test predictions as additional training data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
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
print("PSEUDO-LABELING EXPERIMENT")
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
y_train = train[target].values
test_ids = test['id']

cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
            'study_method', 'facility_rating', 'exam_difficulty']

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[2/6] Preparing data...")

# Label encode for XGBoost/HistGB
X_train_enc = train.drop(['id', target], axis=1).copy()
X_test_enc = test.drop(['id'], axis=1).copy()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train_enc[col])
    X_test_enc[col] = le.transform(X_test_enc[col])
    label_encoders[col] = le

# For CatBoost native
X_train_cat = train.drop(['id', target], axis=1)
X_test_cat = test.drop(['id'], axis=1)
cat_features_idx = [X_train_cat.columns.get_loc(col) for col in cat_cols]

print(f"  Features: {X_train_enc.shape[1]}")

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
    'n_iter_no_change': 50,
}

# ============================================================
# ROUND 1: Generate pseudo-labels
# ============================================================
print("\n[3/6] Round 1: Generating pseudo-labels...")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Arrays for test predictions (will be pseudo-labels)
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_hgb = np.zeros(len(test))

# Also track OOF for baseline comparison
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_hgb = np.zeros(len(train))

X_train_arr = X_train_enc.values
X_test_arr = X_test_enc.values

cat_features_hgb = [X_train_enc.columns.get_loc(col) for col in cat_cols]
HISTGB_PARAMS['categorical_features'] = cat_features_hgb

print("\n  Training base models...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_arr), 1):
    X_tr, X_val = X_train_arr[train_idx], X_train_arr[val_idx]
    X_tr_cat, X_val_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    X_tr_hgb, X_val_hgb = X_train_enc.iloc[train_idx], X_train_enc.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test_arr) / N_FOLDS

    # CatBoost
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_val_cat)
    test_cat += cat_model.predict(X_test_cat) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_hgb, y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_val_hgb)
    test_hgb += hgb_model.predict(X_test_enc) / N_FOLDS

    print(f"    Fold {fold} done")

# Baseline ensemble
blend_baseline = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
rmse_baseline = np.sqrt(np.mean((y_train - blend_baseline) ** 2))

# Pseudo-labels (ensemble of test predictions)
pseudo_labels = 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb
pseudo_labels = np.clip(pseudo_labels, 0, 100)

print(f"\n  Baseline CV RMSE: {rmse_baseline:.5f}")
print(f"\n  Pseudo-label stats:")
print(f"    Min:  {pseudo_labels.min():.2f}")
print(f"    Mean: {pseudo_labels.mean():.2f}")
print(f"    Max:  {pseudo_labels.max():.2f}")

# ============================================================
# ROUND 2: Train with pseudo-labels
# ============================================================
print("\n[4/6] Round 2: Training with pseudo-labels...")

# Create augmented dataset
X_train_aug = pd.concat([X_train_enc, X_test_enc], ignore_index=True)
y_train_aug = np.concatenate([y_train, pseudo_labels])

X_train_cat_aug = pd.concat([X_train_cat, X_test_cat], ignore_index=True)

print(f"\n  Augmented train: {len(X_train_aug):,} samples")
print(f"    Original: {len(train):,}")
print(f"    Pseudo:   {len(test):,}")

# Train on augmented data with 5-fold CV
# Important: Only evaluate on original train data!

oof_xgb_pl = np.zeros(len(train))
oof_cat_pl = np.zeros(len(train))
oof_hgb_pl = np.zeros(len(train))

test_xgb_pl = np.zeros(len(test))
test_cat_pl = np.zeros(len(test))
test_hgb_pl = np.zeros(len(test))

X_train_aug_arr = X_train_aug.values
X_train_aug_hgb = X_train_aug.copy()

print("\n  Training with augmented data...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_arr), 1):
    # Training: use all augmented data EXCEPT validation fold of original train
    # This prevents leakage from original train to validation

    # Indices in augmented dataset
    aug_train_mask = np.ones(len(X_train_aug), dtype=bool)
    aug_train_mask[val_idx] = False  # Exclude original validation

    X_tr_aug = X_train_aug_arr[aug_train_mask]
    y_tr_aug = y_train_aug[aug_train_mask]

    X_tr_cat_aug = X_train_cat_aug.iloc[aug_train_mask]
    X_tr_hgb_aug = X_train_aug_hgb.iloc[aug_train_mask]

    # Validation: only from original train
    X_val = X_train_arr[val_idx]
    X_val_cat = X_train_cat.iloc[val_idx]
    X_val_hgb = X_train_enc.iloc[val_idx]
    y_val = y_train[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb_pl[val_idx] = xgb_model.predict(X_val)
    test_xgb_pl += xgb_model.predict(X_test_arr) / N_FOLDS

    # CatBoost
    train_pool = Pool(X_tr_cat_aug, y_tr_aug, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat_pl[val_idx] = cat_model.predict(X_val_cat)
    test_cat_pl += cat_model.predict(X_test_cat) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_hgb_aug, y_tr_aug)
    oof_hgb_pl[val_idx] = hgb_model.predict(X_val_hgb)
    test_hgb_pl += hgb_model.predict(X_test_enc) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb_pl[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat_pl[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb_pl[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Calculate scores
rmse_xgb_pl = np.sqrt(np.mean((y_train - oof_xgb_pl) ** 2))
rmse_cat_pl = np.sqrt(np.mean((y_train - oof_cat_pl) ** 2))
rmse_hgb_pl = np.sqrt(np.mean((y_train - oof_hgb_pl) ** 2))

blend_pl = 0.30 * oof_xgb_pl + 0.40 * oof_cat_pl + 0.30 * oof_hgb_pl
rmse_pl = np.sqrt(np.mean((y_train - blend_pl) ** 2))

print(f"\n  Results with Pseudo-Labeling:")
print(f"    XGBoost:  {rmse_xgb_pl:.5f}")
print(f"    CatBoost: {rmse_cat_pl:.5f}")
print(f"    HistGB:   {rmse_hgb_pl:.5f}")
print(f"    Ensemble: {rmse_pl:.5f}")

# ============================================================
# ROUND 3: Iterative pseudo-labeling (optional)
# ============================================================
print("\n[5/6] Round 3: Second iteration of pseudo-labeling...")

# Use Round 2 predictions as new pseudo-labels
pseudo_labels_v2 = 0.30 * test_xgb_pl + 0.40 * test_cat_pl + 0.30 * test_hgb_pl
pseudo_labels_v2 = np.clip(pseudo_labels_v2, 0, 100)

# Update augmented dataset
y_train_aug_v2 = np.concatenate([y_train, pseudo_labels_v2])

oof_xgb_pl2 = np.zeros(len(train))
oof_cat_pl2 = np.zeros(len(train))
oof_hgb_pl2 = np.zeros(len(train))

test_xgb_pl2 = np.zeros(len(test))
test_cat_pl2 = np.zeros(len(test))
test_hgb_pl2 = np.zeros(len(test))

print("\n  Training second iteration...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_arr), 1):
    aug_train_mask = np.ones(len(X_train_aug), dtype=bool)
    aug_train_mask[val_idx] = False

    X_tr_aug = X_train_aug_arr[aug_train_mask]
    y_tr_aug = y_train_aug_v2[aug_train_mask]

    X_tr_cat_aug = X_train_cat_aug.iloc[aug_train_mask]
    X_tr_hgb_aug = X_train_aug_hgb.iloc[aug_train_mask]

    X_val = X_train_arr[val_idx]
    X_val_cat = X_train_cat.iloc[val_idx]
    X_val_hgb = X_train_enc.iloc[val_idx]
    y_val = y_train[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb_pl2[val_idx] = xgb_model.predict(X_val)
    test_xgb_pl2 += xgb_model.predict(X_test_arr) / N_FOLDS

    # CatBoost
    train_pool = Pool(X_tr_cat_aug, y_tr_aug, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat_pl2[val_idx] = cat_model.predict(X_val_cat)
    test_cat_pl2 += cat_model.predict(X_test_cat) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_hgb_aug, y_tr_aug)
    oof_hgb_pl2[val_idx] = hgb_model.predict(X_val_hgb)
    test_hgb_pl2 += hgb_model.predict(X_test_enc) / N_FOLDS

    print(f"    Fold {fold} done")

blend_pl2 = 0.30 * oof_xgb_pl2 + 0.40 * oof_cat_pl2 + 0.30 * oof_hgb_pl2
rmse_pl2 = np.sqrt(np.mean((y_train - blend_pl2) ** 2))

print(f"\n  Iteration 2 Ensemble: {rmse_pl2:.5f}")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("[6/6] COMPARISON")
print("=" * 60)

print("\n  Method Comparison:")
print("-" * 50)
print(f"  {'Method':<30} {'CV RMSE':>12}")
print("-" * 50)
print(f"  {'Baseline (no PL)':<30} {rmse_baseline:>12.5f}")
print(f"  {'Pseudo-Label Round 1':<30} {rmse_pl:>12.5f}")
print(f"  {'Pseudo-Label Round 2':<30} {rmse_pl2:>12.5f}")
print("-" * 50)

# Find best
results = {
    'Baseline': (rmse_baseline, 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb),
    'PL Round 1': (rmse_pl, 0.30 * test_xgb_pl + 0.40 * test_cat_pl + 0.30 * test_hgb_pl),
    'PL Round 2': (rmse_pl2, 0.30 * test_xgb_pl2 + 0.40 * test_cat_pl2 + 0.30 * test_hgb_pl2)
}

best_method = min(results.keys(), key=lambda x: results[x][0])
best_rmse, best_pred = results[best_method]

improvement = rmse_baseline - best_rmse
if improvement > 0:
    print(f"\n  BEST: {best_method} with IMPROVEMENT of {improvement:.5f}")
else:
    print(f"\n  NO IMPROVEMENT - Baseline is best")

# Save best submission
predictions = np.clip(best_pred, 0, 100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_pseudo_labeling_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved: {filepath}")

print("\n" + "=" * 60)
