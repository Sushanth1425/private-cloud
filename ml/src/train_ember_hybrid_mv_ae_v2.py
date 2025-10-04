# Base models (LGBM + CatBoost + RF + ExtraTrees) → Stacking → Meta LR


import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib

# ------------------ Config ------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_SAMPLES = 300_000
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load EMBER data ------------------
print("[INFO] Loading EMBER data...")
X = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
y = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)[:len(X)]
y = (y > 127).astype(np.float32)

# ------------------ Subsample ------------------
rng = np.random.default_rng(RANDOM_STATE)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
n_pos = min(len(pos_idx), MAX_SAMPLES // 2)
n_neg = min(len(neg_idx), MAX_SAMPLES // 2)
sample_idx = np.concatenate([rng.choice(pos_idx, n_pos, replace=False),
                             rng.choice(neg_idx, n_neg, replace=False)])
rng.shuffle(sample_idx)
X = X.iloc[sample_idx].astype(np.float32)
y = y[sample_idx]
del pos_idx, neg_idx, sample_idx; gc.collect()

# ------------------ Split train/val/test ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VAL_SPLIT, stratify=y_train, random_state=RANDOM_STATE)
del X_train, y_train, X, y; gc.collect()

# ------------------ Scale features ------------------
scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ------------------ PyTorch Dataset ------------------
train_ds = TensorDataset(torch.from_numpy(X_tr_scaled))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ------------------ Multi-view Autoencoder ------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_ae(X_data, bottleneck, epochs=EPOCHS):
    ae = Autoencoder(input_dim=X_data.shape[1], bottleneck=bottleneck).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        ae.train()
        epoch_loss = 0
        for batch in train_loader:
            xb = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_hat, _ = ae(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"[AE-{bottleneck}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(X_tr_scaled):.6f}")
    return ae

def encode_dataset(model, X_data):
    model.eval()
    z_list = []
    with torch.no_grad():
        for i in range(0, len(X_data), BATCH_SIZE):
            batch = torch.from_numpy(X_data[i:i+BATCH_SIZE]).to(DEVICE)
            _, z = model(batch)
            z_list.append(z.cpu().numpy())
    return np.vstack(z_list)

print("[INFO] Training Multi-view Autoencoders...")
ae1 = train_ae(X_tr_scaled, bottleneck=500)
ae2 = train_ae(X_tr_scaled, bottleneck=300)

print("[INFO] Encoding datasets...")
X_tr_enc1 = encode_dataset(ae1, X_tr_scaled)
X_val_enc1 = encode_dataset(ae1, X_val_scaled)
X_test_enc1 = encode_dataset(ae1, X_test_scaled)

X_tr_enc2 = encode_dataset(ae2, X_tr_scaled)
X_val_enc2 = encode_dataset(ae2, X_val_scaled)
X_test_enc2 = encode_dataset(ae2, X_test_scaled)

# Concatenate multi-view embeddings with original features
X_tr_mv = np.hstack([X_tr_scaled, X_tr_enc1, X_tr_enc2])
X_val_mv = np.hstack([X_val_scaled, X_val_enc1, X_val_enc2])
X_test_mv = np.hstack([X_test_scaled, X_test_enc1, X_test_enc2])

# ------------------ Base Models ------------------
print("[INFO] Initializing base models...")
base_models = [
    ('lgb', lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=128,
        max_depth=12, colsample_bytree=0.8, subsample=0.8,
        class_weight='balanced', random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    ('cat', CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=8, verbose=0,
        class_weights=[1,1], random_state=RANDOM_STATE
    )),
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=12, class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_STATE
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=500, max_depth=12, class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_STATE
    ))
]

meta_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)

stack_clf = StackingClassifier(
    estimators=base_models, final_estimator=meta_model, cv=3, n_jobs=-1, passthrough=True
)

# ------------------ Train Stacking Model ------------------
print("[INFO] Training stacking classifier...")
stack_clf.fit(X_tr_mv, y_tr)

# ------------------ Threshold Tuning ------------------
val_probs = stack_clf.predict_proba(X_val_mv)[:,1]
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_f1 = 0.5, 0
for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"[INFO] Best threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

# ------------------ Evaluate ------------------
test_probs = stack_clf.predict_proba(X_test_mv)[:,1]
test_pred = (test_probs > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_probs)
f1 = f1_score(y_test, test_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------ Save Models ------------------
torch.save(ae1.state_dict(), os.path.join(MODEL_DIR, "ae1.pt"))
torch.save(ae2.state_dict(), os.path.join(MODEL_DIR, "ae2.pt"))
joblib.dump(stack_clf, os.path.join(MODEL_DIR, "stack_hybrid_mv.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(best_thresh, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"\n[DONE] Models + scaler + threshold saved → {MODEL_DIR}")
