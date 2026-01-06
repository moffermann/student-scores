#!/usr/bin/env python3
"""
CatBoost with Native Categorical Features for Playground Series S6E1
"""

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("CATBOOST NATIVE - Student Test Scores")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[2/5] Preparing data...")

# Separate target
target = 'exam_score'
y = train[target]
train_ids = train['id']
test_ids = test['id']

# Drop id and target
X_train = train.drop(['id', target], axis=1)
X_test = test.drop(['id'], axis=1)

# Identify categorical columns (keep as strings for CatBoost)
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Get categorical feature indices
cat_features_idx = [X_train.columns.get_loc(col) for col in cat_cols]

print(f"  Categorical: {cat_cols}")
print(f"  Categorical indices: {cat_features_idx}")
print(f"  Numerical: {num_cols}")
print(f"  Total features: {X_train.shape[1]}")

# ============================================================
# CROSS VALIDATION
# ============================================================
print("\n[3/5] Running 5-Fold Cross Validation...")

# CatBoost parameters
catboost_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.1,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 0,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 50
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Create CatBoost pools
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    cv_scores.append(rmse)
    best_iterations.append(model.get_best_iteration())

    print(f"  Fold {fold}: RMSE = {rmse:.5f} (best iter: {model.get_best_iteration()})")

cv_scores = np.array(cv_scores)
print(f"\n  CV RMSE scores: {cv_scores}")
print(f"  Mean RMSE: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")
print(f"  Mean best iteration: {np.mean(best_iterations):.0f}")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print("\n[4/5] Training final model on all data...")

# Use mean of best iterations
final_iterations = int(np.mean(best_iterations))

final_params = catboost_params.copy()
final_params['iterations'] = final_iterations
final_params.pop('early_stopping_rounds', None)

train_pool = Pool(X_train, y, cat_features=cat_features_idx)
final_model = CatBoostRegressor(**final_params)
final_model.fit(train_pool)

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.get_feature_importance()
}).sort_values('importance', ascending=False)

print("\n  Feature Importance:")
for _, row in importance.iterrows():
    print(f"    {row['feature']:<20} {row['importance']:.2f}")

# ============================================================
# PREDICTIONS & SUBMISSION
# ============================================================
print("\n[5/5] Generating predictions...")

test_pool = Pool(X_test, cat_features=cat_features_idx)
predictions = final_model.predict(test_pool)

# Clip predictions to valid range
predictions = np.clip(predictions, 0, 100)

print(f"\n  Predictions stats (after clipping):")
print(f"    Min:  {predictions.min():.2f}")
print(f"    Mean: {predictions.mean():.2f}")
print(f"    Max:  {predictions.max():.2f}")

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_catboost_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Baseline RMSE:         8.78256")
print(f"  Tuned XGBoost RMSE:    8.76100")
print(f"  CatBoost Native RMSE:  {cv_scores.mean():.5f}")
print(f"  Improvement vs tuned:  {8.76100 - cv_scores.mean():.5f}")
print(f"  Leaderboard top:       ~8.54")
print(f"  Gap to top:            {cv_scores.mean() - 8.54:.3f}")
print("\n" + "=" * 60)
