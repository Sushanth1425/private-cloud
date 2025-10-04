import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib

# ------------------ Config ------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_FEATURES = 300     # Reduce memory usage
MAX_SAMPLES = 100_000  # Subset for RAM
VAL_SPLIT = 0.1
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load EMBER data ------------------
print("[INFO] Loading EMBER data...")
X = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
y = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)[:len(X)]
y = (y > 127).astype(np.float32)
print(f"[INFO] Full data: {X.shape}, positives: {y.mean():.3f}")

# ------------------ Subsample for memory ------------------
rng = np.random.default_rng(RANDOM_STATE)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
n_pos = min(len(pos_idx), MAX_SAMPLES // 2)
n_neg = min(len(neg_idx), MAX_SAMPLES // 2)
sample_idx = np.concatenate([
    rng.choice(pos_idx, n_pos, replace=False),
    rng.choice(neg_idx, n_neg, replace=False)
])
rng.shuffle(sample_idx)

X = X.iloc[sample_idx].copy().astype(np.float32)
y = y[sample_idx]

del pos_idx, neg_idx, sample_idx
gc.collect()

# ------------------ Feature selection ------------------
subset_for_var = X.sample(n=min(50_000, len(X)), random_state=RANDOM_STATE)
variances = subset_for_var.var().sort_values(ascending=False)
top_features = variances.index[:MAX_FEATURES]
X = X[top_features].copy()

# ------------------ Split Train/Val/Test ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SPLIT, stratify=y_train, random_state=RANDOM_STATE
)
del X_train, y_train, X, y
gc.collect()

# ------------------ Scale numeric features ------------------
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------ Convert to PyTorch datasets ------------------
train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ------------------ Define the neural network ------------------
class MalwareNet(nn.Module):
    def __init__(self, input_dim):
        super(MalwareNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = MalwareNet(input_dim=X_tr.shape[1]).to(DEVICE)

# ------------------ Loss and optimizer ------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------ Training loop ------------------
best_val_f1 = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            probs = model(xb)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(yb.cpu().numpy())
    val_preds = np.vstack(val_preds)
    val_labels = np.vstack(val_labels)
    
    # Threshold tuning on validation
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_epoch_f1, best_epoch_thresh = 0, 0.5
    for t in thresholds:
        pred_bin = (val_preds > t).astype(int)
        f1 = f1_score(val_labels, pred_bin)
        if f1 > best_epoch_f1:
            best_epoch_f1, best_epoch_thresh = f1, t

    print(f"Epoch {epoch+1}/{EPOCHS} | Val F1: {best_epoch_f1:.4f} | Threshold: {best_epoch_thresh:.2f}")
    if best_epoch_f1 > best_val_f1:
        best_val_f1 = best_epoch_f1
        best_model_state = model.state_dict()
        best_threshold = best_epoch_thresh

# ------------------ Load best model ------------------
model.load_state_dict(best_model_state)

# ------------------ Evaluate ------------------
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        probs = model(xb)
        test_preds.append(probs.cpu().numpy())
        test_labels.append(yb.cpu().numpy())
test_preds = np.vstack(test_preds)
test_labels = np.vstack(test_labels)
test_pred_bin = (test_preds > best_threshold).astype(int)

acc = accuracy_score(test_labels, test_pred_bin)
auc = roc_auc_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_pred_bin)

print(f"\n✅ Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(test_labels, test_pred_bin, zero_division=0))

# ------------------ Save model, scaler, threshold ------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "nn_model.pt"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(best_threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"\n[DONE] Model + scaler + threshold saved → {MODEL_DIR}")
