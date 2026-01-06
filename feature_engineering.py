#!/usr/bin/env python3
"""
Feature Engineering for Playground Series S6E1: Predicting Student Test Scores
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
print("FEATURE ENGINEERING - Student Test Scores")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n[2/6] Creating new features...")

def create_features(df):
    """Create new features from existing ones."""
    df = df.copy()

    # ---- Interaction Features ----
    # study_hours es la feature más importante (57%)
    df['study_x_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_x_sleep_hours'] = df['study_hours'] * df['sleep_hours']
    df['study_per_age'] = df['study_hours'] / df['age']

    # ---- Ratio Features ----
    df['attendance_per_study'] = df['class_attendance'] / (df['study_hours'] + 1)
    df['sleep_study_ratio'] = df['sleep_hours'] / (df['study_hours'] + 1)

    # ---- Polynomial Features ----
    df['study_hours_sq'] = df['study_hours'] ** 2
    df['attendance_sq'] = df['class_attendance'] ** 2

    # ---- Aggregation Features ----
    # Total effort score
    df['total_effort'] = df['study_hours'] + df['class_attendance']

    # Sleep efficiency (study hours per sleep hour)
    df['sleep_efficiency'] = df['study_hours'] / (df['sleep_hours'] + 1)

    # ---- Binning Features ----
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=5, labels=False)
    df['age_group'] = pd.cut(df['age'], bins=[16, 18, 20, 22, 25], labels=False)

    # ---- Boolean Features ----
    df['high_study'] = (df['study_hours'] > df['study_hours'].median()).astype(int)
    df['high_attendance'] = (df['class_attendance'] > df['class_attendance'].median()).astype(int)
    df['good_sleep'] = (df['sleep_hours'] >= 7).astype(int)

    # ---- Combination Boolean ----
    df['dedicated_student'] = ((df['high_study'] == 1) & (df['high_attendance'] == 1)).astype(int)

    return df

# Apply feature engineering
train = create_features(train)
test = create_features(test)

print(f"  New features created")
print(f"  Train shape: {train.shape}")
print(f"  Test shape:  {test.shape}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n[3/6] Preparing data...")

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

print(f"  Categorical: {len(cat_cols)} features")
print(f"  Numerical: {len(num_cols)} features")

# Label encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print(f"  Total features after encoding: {X_train.shape[1]}")

# List all features
print("\n  Features:")
for i, col in enumerate(X_train.columns, 1):
    print(f"    {i:2d}. {col}")

# ============================================================
# MODEL TRAINING
# ============================================================
print("\n[4/6] Training XGBoost model...")

# Best parameters from hyperparameter tuning
xgb_params = {
    'n_estimators': 485,
    'max_depth': 7,
    'learning_rate': 0.061553,
    'subsample': 0.897410,
    'colsample_bytree': 0.817250,
    'min_child_weight': 10,
    'reg_alpha': 0.003965,
    'reg_lambda': 0.000068,
    'gamma': 0.258847,
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
print("\n[5/6] Training final model on all data...")

model.fit(X_train, y)

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Feature Importance (top 15):")
for i, (_, row) in enumerate(importance.head(15).iterrows(), 1):
    print(f"    {i:2d}. {row['feature']:<25} {row['importance']:.4f}")

# ============================================================
# PREDICTIONS & SUBMISSION
# ============================================================
print("\n[6/6] Generating predictions...")

predictions = model.predict(X_test)

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

filepath = f'{SUBMISSIONS_DIR}/submission_features_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Baseline RMSE:         8.78256")
print(f"  Tuned RMSE:            8.76100")
print(f"  Feature Eng. RMSE:     {rmse_scores.mean():.5f}")
print(f"  Improvement vs tuned:  {8.76100 - rmse_scores.mean():.5f}")
print(f"  Leaderboard top:       ~8.54")
print(f"  Gap to top:            {rmse_scores.mean() - 8.54:.3f}")
print(f"\n  Total features:        {X_train.shape[1]} (was 11)")
print("\n" + "=" * 60)
