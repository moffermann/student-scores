#!/usr/bin/env python3
"""
Baseline Model for Playground Series S6E1: Predicting Student Test Scores
"""

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("BASELINE MODEL - Student Test Scores")
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
# FEATURE ENGINEERING
# ============================================================
print("\n[2/5] Feature engineering...")

# Separate target
target = 'exam_score'
y = train[target]
train_ids = train['id']
test_ids = test['id']

# Drop id and target
X_train = train.drop(['id', target], axis=1)
X_test = test.drop(['id'], axis=1)

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"  Categorical: {cat_cols}")
print(f"  Numerical: {num_cols}")

# Label encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print(f"  Features after encoding: {X_train.shape[1]}")

# ============================================================
# MODEL TRAINING
# ============================================================
print("\n[3/5] Training XGBoost model...")

# XGBoost parameters
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

model = XGBRegressor(**xgb_params)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y, cv=kf, scoring='neg_root_mean_squared_error')
rmse_scores = -cv_scores

print(f"\n  CV RMSE scores: {rmse_scores}")
print(f"  Mean RMSE: {rmse_scores.mean():.5f} (+/- {rmse_scores.std():.5f})")

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print("\n[4/5] Training final model on all data...")

model.fit(X_train, y)

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Feature Importance:")
for _, row in importance.iterrows():
    print(f"    {row['feature']:<20} {row['importance']:.4f}")

# ============================================================
# PREDICTIONS & SUBMISSION
# ============================================================
print("\n[5/5] Generating predictions...")

predictions = model.predict(X_test)

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

filepath = f'{SUBMISSIONS_DIR}/submission_baseline_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  CV RMSE: {rmse_scores.mean():.5f}")
print(f"  Leaderboard top: ~8.54")
print(f"  Gap to top: {rmse_scores.mean() - 8.54:.3f}")
print("\n" + "=" * 60)
