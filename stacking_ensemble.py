#!/usr/bin/env python3
"""
Stacking Ensemble: XGBoost + CatBoost for Playground Series S6E1
Uses out-of-fold predictions to train meta-learner
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("STACKING ENSEMBLE - XGBoost + CatBoost")
print("=" * 60)

# ============================================================
# BEST PARAMETERS FROM PREVIOUS TUNING
# ============================================================

# XGBoost best params (from hyperparameter_tuning.py)
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

# CatBoost best params (from catboost_tuning.py)
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

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/7] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[2/7] Preparing data...")

target = 'exam_score'
y = train[target].values
test_ids = test['id']

# For XGBoost: Label Encoding
X_train_xgb = train.drop(['id', target], axis=1).copy()
X_test_xgb = test.drop(['id'], axis=1).copy()

cat_cols = X_train_xgb.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X_train_xgb[col] = le.fit_transform(X_train_xgb[col])
    X_test_xgb[col] = le.transform(X_test_xgb[col])
    label_encoders[col] = le

X_train_xgb = X_train_xgb.values
X_test_xgb = X_test_xgb.values

# For CatBoost: Native categoricals
X_train_cat = train.drop(['id', target], axis=1)
X_test_cat = test.drop(['id'], axis=1)
cat_features_idx = [X_train_cat.columns.get_loc(col) for col in cat_cols]

print(f"  Features: {X_train_cat.shape[1]}")
print(f"  Categorical: {cat_cols}")

# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================
print("\n[3/7] Generating out-of-fold predictions...")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Arrays for OOF predictions
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

# Arrays for test predictions (averaged over folds)
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

# Track scores
xgb_scores = []
cat_scores = []

print("\n  Training base models with 5-fold CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb), 1):
    print(f"\n  === Fold {fold}/{N_FOLDS} ===")

    # Split data
    X_tr_xgb, X_val_xgb = X_train_xgb[train_idx], X_train_xgb[val_idx]
    X_tr_cat, X_val_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # ----- XGBoost -----
    print("    Training XGBoost...", end=" ")
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False
    )

    # OOF predictions
    oof_xgb[val_idx] = xgb_model.predict(X_val_xgb)
    # Test predictions (average over folds)
    test_xgb += xgb_model.predict(X_test_xgb) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb[val_idx]) ** 2))
    xgb_scores.append(xgb_rmse)
    print(f"RMSE: {xgb_rmse:.5f}")

    # ----- CatBoost -----
    print("    Training CatBoost...", end=" ")
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_idx)

    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # OOF predictions
    oof_cat[val_idx] = cat_model.predict(X_val_cat)
    # Test predictions (average over folds)
    test_cat += cat_model.predict(X_test_cat) / N_FOLDS

    cat_rmse = np.sqrt(np.mean((y_val - oof_cat[val_idx]) ** 2))
    cat_scores.append(cat_rmse)
    print(f"RMSE: {cat_rmse:.5f}")

# ============================================================
# BASE MODEL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("[4/7] Base model CV scores:")
print("=" * 60)

xgb_mean = np.mean(xgb_scores)
cat_mean = np.mean(cat_scores)

print(f"\n  XGBoost:  {xgb_mean:.5f} (+/- {np.std(xgb_scores):.5f})")
print(f"  CatBoost: {cat_mean:.5f} (+/- {np.std(cat_scores):.5f})")

# ============================================================
# BLENDING METHODS
# ============================================================
print("\n" + "=" * 60)
print("[5/7] Testing blending methods...")
print("=" * 60)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Method 1: Simple Average
blend_avg = (oof_xgb + oof_cat) / 2
rmse_avg = rmse(y, blend_avg)
print(f"\n  1. Simple Average:     RMSE = {rmse_avg:.5f}")

# Method 2: Weighted Average (optimize weights)
print("\n  2. Optimizing weights...")
best_rmse = float('inf')
best_w = 0.5

for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_xgb + (1 - w) * oof_cat
    r = rmse(y, blend)
    if r < best_rmse:
        best_rmse = r
        best_w = w

print(f"     Best weight (XGBoost): {best_w:.2f}")
print(f"     Best weight (CatBoost): {1-best_w:.2f}")
print(f"     Weighted Average:   RMSE = {best_rmse:.5f}")

# Method 3: Stacking with Ridge
print("\n  3. Stacking with Ridge meta-learner...")

# Create meta features
meta_train = np.column_stack([oof_xgb, oof_cat])
meta_test = np.column_stack([test_xgb, test_cat])

# Train Ridge on meta features with CV
ridge_scores = []
ridge_oof = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(kf.split(meta_train), 1):
    X_tr_meta, X_val_meta = meta_train[train_idx], meta_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_meta, y_tr)
    ridge_oof[val_idx] = ridge.predict(X_val_meta)

    fold_rmse = rmse(y_val, ridge_oof[val_idx])
    ridge_scores.append(fold_rmse)

rmse_ridge = rmse(y, ridge_oof)
print(f"     Ridge Stacking:     RMSE = {rmse_ridge:.5f} (+/- {np.std(ridge_scores):.5f})")

# Train final Ridge on all data
ridge_final = Ridge(alpha=1.0)
ridge_final.fit(meta_train, y)
print(f"     Ridge coefficients: XGB={ridge_final.coef_[0]:.4f}, CAT={ridge_final.coef_[1]:.4f}")

# ============================================================
# SELECT BEST METHOD
# ============================================================
print("\n" + "=" * 60)
print("[6/7] Selecting best method...")
print("=" * 60)

methods = {
    'Simple Average': (rmse_avg, (oof_xgb + oof_cat) / 2, (test_xgb + test_cat) / 2),
    'Weighted Average': (best_rmse, best_w * oof_xgb + (1-best_w) * oof_cat,
                         best_w * test_xgb + (1-best_w) * test_cat),
    'Ridge Stacking': (rmse_ridge, ridge_oof, ridge_final.predict(meta_test))
}

# Find best
best_method = min(methods.keys(), key=lambda x: methods[x][0])
best_rmse_final, _, best_test_pred = methods[best_method]

print(f"\n  Best method: {best_method}")
print(f"  CV RMSE:     {best_rmse_final:.5f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n  Comparison with previous models:")
print("-" * 50)
print(f"  {'Model':<25} {'CV RMSE':>10} {'vs Best':>10}")
print("-" * 50)
print(f"  {'Baseline':<25} {'8.78256':>10} {'+0.033':>10}")
print(f"  {'XGBoost Tuned':<25} {'8.76100':>10} {'+0.011':>10}")
print(f"  {'CatBoost Tuned':<25} {'8.74974':>10} {'baseline':>10}")
print(f"  {best_method:<25} {best_rmse_final:>10.5f} {best_rmse_final - 8.74974:>+10.5f}")
print("-" * 50)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("[7/7] Generating submission...")
print("=" * 60)

# Clip predictions
predictions = np.clip(best_test_pred, 0, 100)

print(f"\n  Predictions stats:")
print(f"    Min:  {predictions.min():.2f}")
print(f"    Mean: {predictions.mean():.2f}")
print(f"    Max:  {predictions.max():.2f}")

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_stacking_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved: {filepath}")

# Also save individual model predictions for potential further blending
submission_xgb = pd.DataFrame({'id': test_ids, 'exam_score': np.clip(test_xgb, 0, 100)})
submission_cat = pd.DataFrame({'id': test_ids, 'exam_score': np.clip(test_cat, 0, 100)})

submission_xgb.to_csv(f'{SUBMISSIONS_DIR}/submission_xgb_oof_{timestamp}.csv', index=False)
submission_cat.to_csv(f'{SUBMISSIONS_DIR}/submission_cat_oof_{timestamp}.csv', index=False)

print(f"  Also saved individual OOF predictions for further blending")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  XGBoost CV:        {xgb_mean:.5f}")
print(f"  CatBoost CV:       {cat_mean:.5f}")
print(f"  {best_method}:  {best_rmse_final:.5f}")
print(f"\n  Improvement vs CatBoost: {8.74974 - best_rmse_final:.5f}")
print(f"  Gap to top (~8.54):      {best_rmse_final - 8.54:.3f}")

print("\n" + "=" * 60)
