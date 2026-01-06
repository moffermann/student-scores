#!/usr/bin/env python3
"""
Custom Model V2: Profile-Conditioned Deep Regression

Mejoras sobre V1:
1. Procesamiento no-lineal de features numericas
2. FiLM (Feature-wise Linear Modulation): el perfil modula el procesamiento
3. Mayor capacidad en el encoder
4. Residual connections
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Hyperparameters
EMBED_DIM = 16         # Mayor dimension de embeddings
HIDDEN_DIM = 128       # Mayor capacidad
LEARNING_RATE = 0.0005
BATCH_SIZE = 2048
EPOCHS = 100
PATIENCE = 15
N_FOLDS = 5

print("=" * 70)
print("CUSTOM MODEL V2: Profile-Conditioned Deep Regression")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

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
print("\n[2/5] Preprocessing...")

# Label encode categoricals
label_encoders = {}
cat_cardinalities = []

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le
    cat_cardinalities.append(len(le.classes_))

# Standardize numericals
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# Prepare arrays
X_cat_train = train[cat_cols].values.astype(np.int64)
X_num_train = train[num_cols].values.astype(np.float32)
X_cat_test = test[cat_cols].values.astype(np.int64)
X_num_test = test[num_cols].values.astype(np.float32)
y_train = y.astype(np.float32)

print(f"  Categorical shape: {X_cat_train.shape}")
print(f"  Numerical shape: {X_num_train.shape}")

# ============================================================
# DATASET
# ============================================================

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
# FiLM LAYER
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation

    Modula las features x con gamma y beta generados desde conditioning:
    output = gamma * x + beta
    """
    def __init__(self, feature_dim, conditioning_dim):
        super().__init__()
        self.gamma_gen = nn.Linear(conditioning_dim, feature_dim)
        self.beta_gen = nn.Linear(conditioning_dim, feature_dim)

        # Initialize close to identity
        nn.init.ones_(self.gamma_gen.weight.data.mean(dim=0, keepdim=True).T)
        nn.init.zeros_(self.gamma_gen.bias.data)
        nn.init.zeros_(self.beta_gen.weight.data)
        nn.init.zeros_(self.beta_gen.bias.data)

    def forward(self, x, conditioning):
        gamma = self.gamma_gen(conditioning)
        beta = self.beta_gen(conditioning)
        return gamma * x + beta

# ============================================================
# MODEL V2
# ============================================================

class ProfileConditionedDeepRegressor(nn.Module):
    """
    Profile-Conditioned Deep Regression con FiLM

    1. Embeddings de categoricas -> profile vector
    2. FiLM layers modulan el procesamiento de numericas
    3. Multiple capas con residual connections
    """

    def __init__(self, cat_cardinalities, n_num_features, embed_dim=16, hidden_dim=128):
        super().__init__()

        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)

        # Embeddings para cada variable categorica
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])

        total_embed_dim = self.n_cat_features * embed_dim

        # Profile encoder (más profundo)
        self.profile_encoder = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Numerical feature processor con FiLM
        self.num_expand = nn.Linear(n_num_features, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, hidden_dim // 2)

        self.num_hidden1 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film2 = FiLMLayer(hidden_dim, hidden_dim // 2)

        self.num_hidden2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.film3 = FiLMLayer(hidden_dim // 2, hidden_dim // 2)

        # Final layers
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Direct profile contribution
        self.profile_direct = nn.Linear(hidden_dim // 2, 1)

    def forward(self, X_cat, X_num):
        # 1. Get profile embedding
        embeds = [self.embeddings[i](X_cat[:, i]) for i in range(self.n_cat_features)]
        profile_embed = torch.cat(embeds, dim=-1)

        # 2. Encode profile
        profile_vec = self.profile_encoder(profile_embed)

        # 3. Process numerical features with FiLM modulation
        num_h = self.num_expand(X_num)
        num_h = self.film1(num_h, profile_vec)

        num_h = self.num_hidden1(num_h)
        num_h = self.film2(num_h, profile_vec)

        num_h = self.num_hidden2(num_h)
        num_h = self.film3(num_h, profile_vec)

        # 4. Final prediction
        pred_from_num = self.final(num_h).squeeze(-1)

        # 5. Direct profile contribution (baseline per profile)
        pred_from_profile = self.profile_direct(profile_vec).squeeze(-1)

        # 6. Combine
        pred = pred_from_num + pred_from_profile

        return pred

# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_cat, X_num, y in loader:
        X_cat, X_num, y = X_cat.to(DEVICE), X_num.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_cat, X_num)
        loss = criterion(pred, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * len(y)

    return np.sqrt(total_loss / len(loader.dataset))

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    with torch.no_grad():
        for X_cat, X_num, y in loader:
            X_cat, X_num, y = X_cat.to(DEVICE), X_num.to(DEVICE), y.to(DEVICE)
            pred = model(X_cat, X_num)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)
            preds.extend(pred.cpu().numpy())

    return np.sqrt(total_loss / len(loader.dataset)), np.array(preds)

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X_cat, X_num, _ = batch
            else:
                X_cat, X_num = batch
            X_cat, X_num = X_cat.to(DEVICE), X_num.to(DEVICE)
            pred = model(X_cat, X_num)
            preds.extend(pred.cpu().numpy())
    return np.array(preds)

# ============================================================
# CROSS-VALIDATION
# ============================================================
print("\n[3/5] Training with Cross-Validation...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_cat_train), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}")

    # Split data
    X_cat_tr, X_cat_val = X_cat_train[train_idx], X_cat_train[val_idx]
    X_num_tr, X_num_val = X_num_train[train_idx], X_num_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Create datasets
    train_dataset = StudentDataset(X_cat_tr, X_num_tr, y_tr)
    val_dataset = StudentDataset(X_cat_val, X_num_val, y_val)
    test_dataset = StudentDataset(X_cat_test, X_num_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Create model
    model = ProfileConditionedDeepRegressor(
        cat_cardinalities=cat_cardinalities,
        n_num_features=len(num_cols),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop with early stopping
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_rmse = train_epoch(model, train_loader, optimizer, criterion)
        val_rmse, val_preds_epoch = validate(model, val_loader, criterion)
        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}: train={train_rmse:.5f}, val={val_rmse:.5f}, lr={lr:.6f}")

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    _, oof_preds[val_idx] = validate(model, val_loader, criterion)
    test_preds += predict(model, test_loader) / N_FOLDS

    fold_scores.append(best_val_rmse)
    print(f"    Best Val RMSE: {best_val_rmse:.5f}")

# ============================================================
# RESULTS
# ============================================================
print("\n[4/5] Results...")

overall_rmse = np.sqrt(np.mean((y_train - oof_preds) ** 2))

print(f"\n  Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
print(f"  Mean fold RMSE: {np.mean(fold_scores):.5f}")
print(f"  Overall CV RMSE: {overall_rmse:.5f}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
print("\n[5/5] Saving submission...")

predictions = np.clip(test_preds, 0, 100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_custom_model_v2_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved: {filepath}")

print(f"\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n  Custom Model V2 CV RMSE: {overall_rmse:.5f}")
print(f"  Best baseline (pseudo-labeling): 8.72348")
print(f"  Difference: {overall_rmse - 8.72348:+.5f}")

print("\n" + "=" * 70)
