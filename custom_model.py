#!/usr/bin/env python3
"""
Custom Model: Profile-Conditioned Regression

Arquitectura:
    y = (β₀ + α_profile) + (β + γ_profile) · x_num

Donde:
    - β₀ = intercepto global (learnable)
    - β = pendientes globales para features numericas (4 valores)
    - α_profile = ajuste de intercepto por perfil (generado desde embedding)
    - γ_profile = ajuste de pendientes por perfil (generado desde embedding)

El perfil se representa como embeddings de cada variable categorica,
que luego se combinan para generar los ajustes.
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
EMBED_DIM = 8          # Dimension de embeddings por categoria
HIDDEN_DIM = 64        # Dimension de capa oculta
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 50
PATIENCE = 10
N_FOLDS = 5

print("=" * 70)
print("CUSTOM MODEL: Profile-Conditioned Regression")
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
    print(f"  {col}: {len(le.classes_)} classes")

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

print(f"\n  Categorical shape: {X_cat_train.shape}")
print(f"  Numerical shape: {X_num_train.shape}")
print(f"  Cardinalities: {cat_cardinalities}")

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
# MODEL
# ============================================================

class ProfileConditionedRegressor(nn.Module):
    """
    Profile-Conditioned Regression Network

    y = (β₀ + α_profile) + (β + γ_profile) · x_num

    Donde:
    - β₀, β son parametros globales
    - α_profile, γ_profile son generados desde el embedding del perfil
    """

    def __init__(self, cat_cardinalities, n_num_features, embed_dim=8, hidden_dim=64):
        super().__init__()

        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)

        # Embeddings para cada variable categorica
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])

        # Total dimension after concatenating all embeddings
        total_embed_dim = self.n_cat_features * embed_dim

        # Profile encoder: embeddings concatenados -> vector de perfil
        self.profile_encoder = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Generador de coeficientes: perfil -> (intercepto, pendientes)
        # Output: 1 (intercepto) + n_num_features (pendientes)
        self.coef_generator = nn.Linear(hidden_dim // 2, 1 + n_num_features)

        # Parametros GLOBALES (baseline)
        self.global_intercept = nn.Parameter(torch.tensor(62.0))  # ~media del target
        self.global_slopes = nn.Parameter(torch.zeros(n_num_features))

        # Scaling factor para los ajustes (empieza pequeño)
        self.adjustment_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, X_cat, X_num):
        batch_size = X_cat.size(0)

        # 1. Obtener embeddings de cada categoria
        embeds = [self.embeddings[i](X_cat[:, i]) for i in range(self.n_cat_features)]
        profile_embed = torch.cat(embeds, dim=-1)  # [batch, total_embed_dim]

        # 2. Encodear perfil
        profile_vec = self.profile_encoder(profile_embed)  # [batch, hidden_dim//2]

        # 3. Generar coeficientes condicionados al perfil
        coefs = self.coef_generator(profile_vec)  # [batch, 1 + n_num]

        alpha_profile = coefs[:, 0]  # Ajuste de intercepto [batch]
        gamma_profile = coefs[:, 1:]  # Ajuste de pendientes [batch, n_num]

        # 4. Calcular prediccion
        # Intercepto: global + ajuste_perfil
        intercept = self.global_intercept + self.adjustment_scale * alpha_profile

        # Pendientes: global + ajuste_perfil
        slopes = self.global_slopes.unsqueeze(0) + self.adjustment_scale * gamma_profile

        # Prediccion lineal condicionada
        linear_term = (slopes * X_num).sum(dim=-1)  # [batch]

        pred = intercept + linear_term

        return pred

    def get_profile_coefficients(self, X_cat):
        """Extraer los coeficientes para un perfil dado (para interpretabilidad)"""
        with torch.no_grad():
            embeds = [self.embeddings[i](X_cat[:, i]) for i in range(self.n_cat_features)]
            profile_embed = torch.cat(embeds, dim=-1)
            profile_vec = self.profile_encoder(profile_embed)
            coefs = self.coef_generator(profile_vec)

            alpha = coefs[:, 0] * self.adjustment_scale
            gamma = coefs[:, 1:] * self.adjustment_scale

            final_intercept = self.global_intercept + alpha
            final_slopes = self.global_slopes.unsqueeze(0) + gamma

            return final_intercept, final_slopes

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Create model
    model = ProfileConditionedRegressor(
        cat_cardinalities=cat_cardinalities,
        n_num_features=len(num_cols),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_rmse = train_epoch(model, train_loader, optimizer, criterion)
        val_rmse, val_preds = validate(model, val_loader, criterion)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train={train_rmse:.5f}, val={val_rmse:.5f}")

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
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
# INTERPRETABILITY
# ============================================================
print("\n[5/5] Model Interpretability...")

# Load last model for inspection
model.eval()

print(f"\n  Global parameters:")
print(f"    Intercept: {model.global_intercept.item():.4f}")
print(f"    Slopes: {model.global_slopes.detach().cpu().numpy()}")
print(f"    Adjustment scale: {model.adjustment_scale.item():.4f}")

# Sample some profiles and show their coefficients
print(f"\n  Sample profile coefficients:")
sample_idx = [0, 1000, 5000, 10000]
sample_cats = torch.LongTensor(X_cat_train[sample_idx]).to(DEVICE)
intercepts, slopes = model.get_profile_coefficients(sample_cats)

for i, idx in enumerate(sample_idx):
    profile = "_".join([str(X_cat_train[idx, j]) for j in range(len(cat_cols))])
    print(f"\n    Profile {idx} ({profile}):")
    print(f"      Intercept: {intercepts[i].item():.4f}")
    print(f"      Slopes: {slopes[i].detach().cpu().numpy()}")

# ============================================================
# SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("SAVING SUBMISSION")
print("=" * 70)

predictions = np.clip(test_preds, 0, 100)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': predictions
})

filepath = f'{SUBMISSIONS_DIR}/submission_custom_model_{timestamp}.csv'
submission.to_csv(filepath, index=False)
print(f"\n  Saved: {filepath}")

print(f"\n  Custom Model CV RMSE: {overall_rmse:.5f}")
print(f"  Best baseline (pseudo-labeling): 8.72348")
print(f"  Difference: {overall_rmse - 8.72348:+.5f}")

print("\n" + "=" * 70)
