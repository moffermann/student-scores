#!/usr/bin/env python3
"""
Hyperparameter Tuning for Playground Series S6E1: Predicting Student Test Scores
Using Optuna for Bayesian optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("HYPERPARAMETER TUNING - Student Test Scores")
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
print("\n[2/6] Feature engineering...")

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
# OPTUNA OPTIMIZATION
# ============================================================
print("\n[3/6] Running Optuna optimization...")
print("  This may take several minutes...")

N_TRIALS = 50
CV_FOLDS = 5

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    model = XGBRegressor(**params)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y, cv=kf, scoring='neg_root_mean_squared_error')

    return -cv_scores.mean()

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n  Completed {N_TRIALS} trials")
print(f"  Best RMSE: {study.best_value:.5f}")

# ============================================================
# BEST PARAMETERS
# ============================================================
print("\n[4/6] Best hyperparameters found:")

best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbosity'] = 0

for param, value in best_params.items():
    if isinstance(value, float):
        print(f"    {param}: {value:.6f}")
    else:
        print(f"    {param}: {value}")

# ============================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================
print("\n[5/6] Training final model with best parameters...")

model = XGBRegressor(**best_params)

# Cross-validation with best params
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y, cv=kf, scoring='neg_root_mean_squared_error')
rmse_scores = -cv_scores

print(f"\n  CV RMSE scores: {rmse_scores}")
print(f"  Mean RMSE: {rmse_scores.mean():.5f} (+/- {rmse_scores.std():.5f})")

# Train on all data
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

filepath = f'{SUBMISSIONS_DIR}/submission_tuned_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Baseline RMSE:    8.78256")
print(f"  Tuned RMSE:       {rmse_scores.mean():.5f}")
print(f"  Improvement:      {8.78256 - rmse_scores.mean():.5f}")
print(f"  Leaderboard top:  ~8.54")
print(f"  Gap to top:       {rmse_scores.mean() - 8.54:.3f}")
print("\n" + "=" * 60)

# ============================================================
# SAVE BEST PARAMS FOR REFERENCE
# ============================================================
print("\n  Best params dict for copy/paste:")
print(f"  {best_params}")
