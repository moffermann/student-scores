#!/usr/bin/env python3
"""
Stacking Ensemble: XGBoost + CatBoost + HistGradientBoosting for Playground Series S6E1
Includes quick Optuna tuning for HistGB, then 3-model ensemble
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
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("STACKING ENSEMBLE - XGBoost + CatBoost + HistGradientBoosting")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/8] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[2/8] Preparing data...")

target = 'exam_score'
y = train[target].values
test_ids = test['id']

# For XGBoost/HistGB: Label Encoding
X_train_enc = train.drop(['id', target], axis=1).copy()
X_test_enc = test.drop(['id'], axis=1).copy()

cat_cols = X_train_enc.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train_enc[col])
    X_test_enc[col] = le.transform(X_test_enc[col])
    label_encoders[col] = le

X_train_xgb = X_train_enc.values
X_test_xgb = X_test_enc.values

# For HistGB: use encoded DataFrame
X_train_hgb = X_train_enc.copy()
X_test_hgb = X_test_enc.copy()
# Mark categorical features for HistGB
cat_features_hgb = [X_train_hgb.columns.get_loc(col) for col in cat_cols]

# For CatBoost: Native categoricals
X_train_cat = train.drop(['id', target], axis=1)
X_test_cat = test.drop(['id'], axis=1)
cat_features_idx = [X_train_cat.columns.get_loc(col) for col in cat_cols]

print(f"  Features: {X_train_cat.shape[1]}")
print(f"  Categorical: {cat_cols}")

# ============================================================
# BEST PARAMETERS (XGBoost & CatBoost from previous tuning)
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

# ============================================================
# HISTGRADIENTBOOSTING TUNING WITH OPTUNA
# ============================================================
print("\n[3/8] Tuning HistGradientBoosting with Optuna (30 trials)...")

N_TRIALS_HGB = 30
CV_FOLDS = 5

def hgb_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255),
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'categorical_features': cat_features_hgb
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_train_hgb):
        X_tr = X_train_hgb.iloc[train_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        X_val = X_train_hgb.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        cv_scores.append(rmse)

    return np.mean(cv_scores)

sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(hgb_objective, n_trials=N_TRIALS_HGB, show_progress_bar=True)

print(f"\n  Best HistGB RMSE: {study.best_value:.5f}")
print(f"  Best trial: {study.best_trial.number}")

# Build HistGB params
HISTGB_PARAMS = study.best_params.copy()
HISTGB_PARAMS['random_state'] = 42
HISTGB_PARAMS['early_stopping'] = True
HISTGB_PARAMS['validation_fraction'] = 0.1
HISTGB_PARAMS['n_iter_no_change'] = 50
HISTGB_PARAMS['categorical_features'] = cat_features_hgb

print("\n  Best HistGB params:")
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f"    {k}: {v:.6f}")
    else:
        print(f"    {k}: {v}")

# ============================================================
# OUT-OF-FOLD PREDICTIONS (3 models)
# ============================================================
print("\n[4/8] Generating out-of-fold predictions for 3 models...")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Arrays for OOF predictions
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_hgb = np.zeros(len(train))

# Arrays for test predictions
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_hgb = np.zeros(len(test))

# Track scores
xgb_scores = []
cat_scores = []
hgb_scores = []

print("\n  Training base models with 5-fold CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb), 1):
    print(f"\n  === Fold {fold}/{N_FOLDS} ===")

    # Split data
    X_tr_xgb, X_val_xgb = X_train_xgb[train_idx], X_train_xgb[val_idx]
    X_tr_cat, X_val_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    X_tr_hgb, X_val_hgb = X_train_hgb.iloc[train_idx], X_train_hgb.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # ----- XGBoost -----
    print("    Training XGBoost...", end=" ")
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)

    oof_xgb[val_idx] = xgb_model.predict(X_val_xgb)
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

    oof_cat[val_idx] = cat_model.predict(X_val_cat)
    test_cat += cat_model.predict(X_test_cat) / N_FOLDS

    cat_rmse = np.sqrt(np.mean((y_val - oof_cat[val_idx]) ** 2))
    cat_scores.append(cat_rmse)
    print(f"RMSE: {cat_rmse:.5f}")

    # ----- HistGradientBoosting -----
    print("    Training HistGB...", end=" ")
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr_hgb, y_tr)

    oof_hgb[val_idx] = hgb_model.predict(X_val_hgb)
    test_hgb += hgb_model.predict(X_test_hgb) / N_FOLDS

    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb[val_idx]) ** 2))
    hgb_scores.append(hgb_rmse)
    print(f"RMSE: {hgb_rmse:.5f}")

# ============================================================
# BASE MODEL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("[5/8] Base model CV scores:")
print("=" * 60)

xgb_mean = np.mean(xgb_scores)
cat_mean = np.mean(cat_scores)
hgb_mean = np.mean(hgb_scores)

print(f"\n  XGBoost:  {xgb_mean:.5f} (+/- {np.std(xgb_scores):.5f})")
print(f"  CatBoost: {cat_mean:.5f} (+/- {np.std(cat_scores):.5f})")
print(f"  HistGB:   {hgb_mean:.5f} (+/- {np.std(hgb_scores):.5f})")

# ============================================================
# BLENDING METHODS
# ============================================================
print("\n" + "=" * 60)
print("[6/8] Testing blending methods...")
print("=" * 60)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Method 1: Simple Average (3 models)
blend_avg = (oof_xgb + oof_cat + oof_hgb) / 3
rmse_avg = rmse(y, blend_avg)
print(f"\n  1. Simple Average (3 models): RMSE = {rmse_avg:.5f}")

# Method 2: Optimize weights with grid search
print("\n  2. Optimizing 3-model weights...")
best_rmse = float('inf')
best_weights = (1/3, 1/3, 1/3)

for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        blend = w1 * oof_xgb + w2 * oof_cat + w3 * oof_hgb
        r = rmse(y, blend)
        if r < best_rmse:
            best_rmse = r
            best_weights = (w1, w2, w3)

print(f"     Best weights: XGB={best_weights[0]:.2f}, CAT={best_weights[1]:.2f}, HGB={best_weights[2]:.2f}")
print(f"     Weighted Average:   RMSE = {best_rmse:.5f}")

# Method 3: Ridge Stacking
print("\n  3. Stacking with Ridge meta-learner...")

meta_train = np.column_stack([oof_xgb, oof_cat, oof_hgb])
meta_test = np.column_stack([test_xgb, test_cat, test_hgb])

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

ridge_final = Ridge(alpha=1.0)
ridge_final.fit(meta_train, y)
print(f"     Ridge coefficients: XGB={ridge_final.coef_[0]:.4f}, CAT={ridge_final.coef_[1]:.4f}, HGB={ridge_final.coef_[2]:.4f}")

# Method 4: Best 2-model blend (for comparison)
print("\n  4. Best 2-model combinations:")

# XGB + CAT
best_w_xc = 0.43
blend_xc = best_w_xc * oof_xgb + (1-best_w_xc) * oof_cat
rmse_xc = rmse(y, blend_xc)
print(f"     XGB + CAT (0.43/0.57): RMSE = {rmse_xc:.5f}")

# XGB + HGB
best_rmse_xh = float('inf')
best_w_xh = 0.5
for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_xgb + (1-w) * oof_hgb
    r = rmse(y, blend)
    if r < best_rmse_xh:
        best_rmse_xh = r
        best_w_xh = w
print(f"     XGB + HGB ({best_w_xh:.2f}/{1-best_w_xh:.2f}): RMSE = {best_rmse_xh:.5f}")

# CAT + HGB
best_rmse_ch = float('inf')
best_w_ch = 0.5
for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_cat + (1-w) * oof_hgb
    r = rmse(y, blend)
    if r < best_rmse_ch:
        best_rmse_ch = r
        best_w_ch = w
print(f"     CAT + HGB ({best_w_ch:.2f}/{1-best_w_ch:.2f}): RMSE = {best_rmse_ch:.5f}")

# ============================================================
# SELECT BEST METHOD
# ============================================================
print("\n" + "=" * 60)
print("[7/8] Selecting best method...")
print("=" * 60)

methods = {
    'Simple Avg (3)': (rmse_avg, (test_xgb + test_cat + test_hgb) / 3),
    'Weighted Avg (3)': (best_rmse, best_weights[0]*test_xgb + best_weights[1]*test_cat + best_weights[2]*test_hgb),
    'Ridge Stacking (3)': (rmse_ridge, ridge_final.predict(meta_test)),
    'XGB+CAT (2)': (rmse_xc, best_w_xc*test_xgb + (1-best_w_xc)*test_cat),
    'XGB+HGB (2)': (best_rmse_xh, best_w_xh*test_xgb + (1-best_w_xh)*test_hgb),
    'CAT+HGB (2)': (best_rmse_ch, best_w_ch*test_cat + (1-best_w_ch)*test_hgb),
}

print("\n  All methods comparison:")
print("-" * 50)
for name, (score, _) in sorted(methods.items(), key=lambda x: x[1][0]):
    print(f"    {name:<20} RMSE: {score:.5f}")
print("-" * 50)

best_method = min(methods.keys(), key=lambda x: methods[x][0])
best_rmse_final, best_test_pred = methods[best_method]

print(f"\n  Best method: {best_method}")
print(f"  CV RMSE:     {best_rmse_final:.5f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n  Comparison with previous models:")
print("-" * 55)
print(f"  {'Model':<30} {'CV RMSE':>10} {'vs Best':>10}")
print("-" * 55)
print(f"  {'Baseline':<30} {'8.78256':>10} {'+0.045':>10}")
print(f"  {'XGBoost Tuned':<30} {'8.76100':>10} {'+0.023':>10}")
print(f"  {'CatBoost Tuned':<30} {'8.74974':>10} {'+0.012':>10}")
print(f"  {'2-Model Stacking (prev)':<30} {'8.73771':>10} {'baseline':>10}")
print(f"  {best_method:<30} {best_rmse_final:>10.5f} {best_rmse_final - 8.73771:>+10.5f}")
print("-" * 55)

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("[8/8] Generating submission...")
print("=" * 60)

predictions = np.clip(best_test_pred, 0, 100)

print(f"\n  Predictions stats:")
print(f"    Min:  {predictions.min():.2f}")
print(f"    Mean: {predictions.mean():.2f}")
print(f"    Max:  {predictions.max():.2f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_stacking3_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  Individual model scores:")
print(f"    XGBoost:  {xgb_mean:.5f}")
print(f"    CatBoost: {cat_mean:.5f}")
print(f"    HistGB:   {hgb_mean:.5f}")

print(f"\n  Best ensemble: {best_method}")
print(f"  CV RMSE:       {best_rmse_final:.5f}")
print(f"\n  Improvement vs 2-model: {8.73771 - best_rmse_final:.5f}")
print(f"  Gap to top (~8.54):     {best_rmse_final - 8.54:.3f}")

print("\n  HistGB best params for reference:")
print(f"  {HISTGB_PARAMS}")

print("\n" + "=" * 60)
