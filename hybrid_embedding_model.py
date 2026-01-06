#!/usr/bin/env python3
"""
Hybrid Model: Neural Network Embeddings + Tree Ensemble

Estrategia:
1. Entrenar una red simple para aprender embeddings de categoricas
2. Extraer los embeddings
3. Usar embeddings + numericas como features para XGBoost/CatBoost
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
SUBMISSIONS_DIR = '/home/mauro/kaggle/student-scores/submissions'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EMBED_DIM = 4  # Dimension de embeddings
N_FOLDS = 5

print("=" * 70)
print("HYBRID MODEL: Embeddings + Tree Ensemble")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")

target = 'exam_score'
cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality',
            'study_method', 'facility_rating', 'exam_difficulty']
num_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'age']

y = train[target].values
test_ids = test['id']

# ============================================================
# PREPROCESSING
# ============================================================
print("\n[2/6] Preprocessing...")

# Label encode categoricals
label_encoders = {}
cat_cardinalities = []

train_cat_encoded = train[cat_cols].copy()
test_cat_encoded = test[cat_cols].copy()

for col in cat_cols:
    le = LabelEncoder()
    train_cat_encoded[col] = le.fit_transform(train[col])
    test_cat_encoded[col] = le.transform(test[col])
    label_encoders[col] = le
    cat_cardinalities.append(len(le.classes_))

# Keep original numericals
train_num = train[num_cols].values.astype(np.float32)
test_num = test[num_cols].values.astype(np.float32)

X_cat_train = train_cat_encoded.values.astype(np.int64)
X_cat_test = test_cat_encoded.values.astype(np.int64)
y_train = y.astype(np.float32)

print(f"  Cardinalities: {dict(zip(cat_cols, cat_cardinalities))}")

# ============================================================
# SIMPLE EMBEDDING MODEL
# ============================================================

class EmbeddingModel(nn.Module):
    """Simple model to learn embeddings"""

    def __init__(self, cat_cardinalities, n_num_features, embed_dim=4):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])

        total_embed_dim = len(cat_cardinalities) * embed_dim
        input_dim = total_embed_dim + n_num_features

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X_cat, X_num):
        embeds = [self.embeddings[i](X_cat[:, i]) for i in range(len(self.embeddings))]
        cat_embed = torch.cat(embeds, dim=-1)
        x = torch.cat([cat_embed, X_num], dim=-1)
        return self.fc(x).squeeze(-1)

    def get_embeddings(self, X_cat):
        """Extract embeddings for each sample"""
        with torch.no_grad():
            embeds = [self.embeddings[i](X_cat[:, i]) for i in range(len(self.embeddings))]
            return torch.cat(embeds, dim=-1)

class StudentDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_num = torch.FloatTensor(X_num)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]

# ============================================================
# TRAIN EMBEDDING MODEL
# ============================================================
print("\n[3/6] Training embedding model...")

# Standardize numericals for NN
scaler_nn = StandardScaler()
train_num_scaled = scaler_nn.fit_transform(train_num)
test_num_scaled = scaler_nn.transform(test_num)

# Create dataset
train_dataset = StudentDataset(X_cat_train, train_num_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

# Train model
model = EmbeddingModel(
    cat_cardinalities=cat_cardinalities,
    n_num_features=len(num_cols),
    embed_dim=EMBED_DIM
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_cat, X_num, y in train_loader:
        X_cat, X_num, y = X_cat.to(DEVICE), X_num.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_cat, X_num)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    rmse = np.sqrt(total_loss / len(train_dataset))
    if (epoch + 1) % 5 == 0:
        print(f"    Epoch {epoch+1}: RMSE={rmse:.5f}")

# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================
print("\n[4/6] Extracting embeddings...")

model.eval()

# Get embeddings for train
train_cat_tensor = torch.LongTensor(X_cat_train).to(DEVICE)
train_embeddings = model.get_embeddings(train_cat_tensor).cpu().numpy()

# Get embeddings for test
test_cat_tensor = torch.LongTensor(X_cat_test).to(DEVICE)
test_embeddings = model.get_embeddings(test_cat_tensor).cpu().numpy()

print(f"  Train embeddings shape: {train_embeddings.shape}")
print(f"  Test embeddings shape: {test_embeddings.shape}")

# Create embedding column names
embed_cols = []
for i, col in enumerate(cat_cols):
    for j in range(EMBED_DIM):
        embed_cols.append(f'{col}_emb_{j}')

print(f"  Embedding features: {len(embed_cols)}")

# ============================================================
# PREPARE FEATURES FOR TREES
# ============================================================
print("\n[5/6] Preparing features for tree models...")

# Combine: embeddings + numerical features
X_train_hybrid = np.hstack([train_embeddings, train_num])
X_test_hybrid = np.hstack([test_embeddings, test_num])

feature_names = embed_cols + num_cols
print(f"  Total features: {len(feature_names)}")
print(f"  X_train shape: {X_train_hybrid.shape}")

# ============================================================
# TREE MODEL PARAMS
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
# TRAIN TREE ENSEMBLE
# ============================================================
print("\n[6/6] Training tree ensemble with embeddings...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_hgb = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_hgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_hybrid), 1):
    X_tr, X_val = X_train_hybrid[train_idx], X_train_hybrid[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test_hybrid) / N_FOLDS

    # CatBoost
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat += cat_model.predict(X_test_hybrid) / N_FOLDS

    # HistGB
    hgb_model = HistGradientBoostingRegressor(**HISTGB_PARAMS)
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_val)
    test_hgb += hgb_model.predict(X_test_hybrid) / N_FOLDS

    xgb_rmse = np.sqrt(np.mean((y_val - oof_xgb[val_idx]) ** 2))
    cat_rmse = np.sqrt(np.mean((y_val - oof_cat[val_idx]) ** 2))
    hgb_rmse = np.sqrt(np.mean((y_val - oof_hgb[val_idx]) ** 2))
    print(f"    Fold {fold}: XGB={xgb_rmse:.5f}, CAT={cat_rmse:.5f}, HGB={hgb_rmse:.5f}")

# Calculate overall scores
rmse_xgb = np.sqrt(np.mean((y_train - oof_xgb) ** 2))
rmse_cat = np.sqrt(np.mean((y_train - oof_cat) ** 2))
rmse_hgb = np.sqrt(np.mean((y_train - oof_hgb) ** 2))

# Ensemble
blend = 0.30 * oof_xgb + 0.40 * oof_cat + 0.30 * oof_hgb
rmse_ensemble = np.sqrt(np.mean((y_train - blend) ** 2))

test_blend = 0.30 * test_xgb + 0.40 * test_cat + 0.30 * test_hgb

print(f"\n  Results:")
print(f"    XGBoost:  {rmse_xgb:.5f}")
print(f"    CatBoost: {rmse_cat:.5f}")
print(f"    HistGB:   {rmse_hgb:.5f}")
print(f"    Ensemble: {rmse_ensemble:.5f}")

# Save
predictions = np.clip(test_blend, 0, 100)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({'id': test_ids, 'exam_score': predictions})
filepath = f'{SUBMISSIONS_DIR}/submission_hybrid_embedding_{timestamp}.csv'
submission.to_csv(filepath, index=False)

print(f"\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n  Hybrid (Embeddings + Trees): {rmse_ensemble:.5f}")
print(f"  Best baseline (pseudo-labeling): 8.72348")
print(f"  Difference: {rmse_ensemble - 8.72348:+.5f}")
print(f"\n  Saved: {filepath}")
print("=" * 70)
