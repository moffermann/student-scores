#!/usr/bin/env python3
"""
CatBoost Hyperparameter Tuning with Optuna for Playground Series S6E1
"""

import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

print("=" * 60)
print("CATBOOST TUNING - Student Test Scores")
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
# PREPARE DATA
# ============================================================
print("\n[2/6] Preparing data...")

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
cat_features_idx = [X_train.columns.get_loc(col) for col in cat_cols]

print(f"  Categorical: {cat_cols}")
print(f"  Total features: {X_train.shape[1]}")

# ============================================================
# OPTUNA OPTIMIZATION
# ============================================================
print("\n[3/6] Running Optuna optimization...")
print("  This may take several minutes...")

N_TRIALS = 50
CV_FOLDS = 5

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 50,
        'task_type': 'GPU',
        'devices': '0'
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        cv_scores.append(rmse)

    return np.mean(cv_scores)

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
best_params['random_seed'] = 42
best_params['verbose'] = 0
best_params['loss_function'] = 'RMSE'
best_params['eval_metric'] = 'RMSE'
best_params['task_type'] = 'GPU'
best_params['devices'] = '0'

for param, value in best_params.items():
    if isinstance(value, float):
        print(f"    {param}: {value:.6f}")
    else:
        print(f"    {param}: {value}")

# ============================================================
# FINAL CV WITH BEST PARAMS
# ============================================================
print("\n[5/6] Final CV with best parameters...")

best_params['early_stopping_rounds'] = 50

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

    model = CatBoostRegressor(**best_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    cv_scores.append(rmse)
    best_iterations.append(model.get_best_iteration())

    print(f"  Fold {fold}: RMSE = {rmse:.5f} (best iter: {model.get_best_iteration()})")

cv_scores = np.array(cv_scores)
print(f"\n  CV RMSE scores: {cv_scores}")
print(f"  Mean RMSE: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")

# Train final model
print("\n  Training final model on all data...")

final_iterations = int(np.mean(best_iterations))
final_params = best_params.copy()
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
print("\n[6/6] Generating predictions...")

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

filepath = f'{SUBMISSIONS_DIR}/submission_catboost_tuned_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Baseline RMSE:            8.78256")
print(f"  Tuned XGBoost RMSE:       8.76100")
print(f"  CatBoost Default RMSE:    8.76817")
print(f"  CatBoost Tuned RMSE:      {cv_scores.mean():.5f}")
print(f"  Improvement vs XGBoost:   {8.76100 - cv_scores.mean():.5f}")
print(f"  Leaderboard top:          ~8.54")
print(f"  Gap to top:               {cv_scores.mean() - 8.54:.3f}")
print("\n" + "=" * 60)

# Save best params for reference
print("\n  Best params dict for copy/paste:")
print(f"  {best_params}")
