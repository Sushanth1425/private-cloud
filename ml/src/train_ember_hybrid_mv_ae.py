# ember_hybrid_mv_ae.py
import os, gc, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import catboost as cb
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
AE_BOTTLENECK = 512
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
sample_idx = np.concatenate([
    rng.choice(pos_idx, n_pos, replace=False),
    rng.choice(neg_idx, n_neg, replace=False)
])
rng.shuffle(sample_idx)
X = X.iloc[sample_idx].astype(np.float32)
y = y[sample_idx]
del pos_idx, neg_idx, sample_idx; gc.collect()

# ------------------ Split train/val/test ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SPLIT, stratify=y_train, random_state=RANDOM_STATE)
del X_train, y_train, X, y; gc.collect()

# ------------------ Scale features ------------------
scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ------------------ Multi-view AE ------------------
class MultiViewAE(nn.Module):
    def __init__(self, input_dim, bottleneck=512):
        super().__init__()
        self.encoder_raw = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, bottleneck),
            nn.ReLU()
        )
        self.encoder_log = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, bottleneck),
            nn.ReLU()
        )
        self.decoder_raw = nn.Sequential(
            nn.Linear(bottleneck, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, input_dim)
        )
        self.decoder_log = nn.Sequential(
            nn.Linear(bottleneck, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        z_raw = self.encoder_raw(x)
        z_log = self.encoder_log(torch.log1p(x))
        x_hat_raw = self.decoder_raw(z_raw)
        x_hat_log = self.decoder_log(z_log)
        z_comb = torch.cat([z_raw, z_log], dim=1)
        return x_hat_raw, x_hat_log, z_comb

# Add Gaussian noise
def add_noise(x, noise_level=0.05):
    return x + noise_level * np.random.normal(size=x.shape)

train_ds = TensorDataset(torch.from_numpy(add_noise(X_tr_scaled)).float())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

ae = MultiViewAE(input_dim=X_tr_scaled.shape[1], bottleneck=AE_BOTTLENECK).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)

print("[INFO] Training Multi-View Autoencoder...")
for epoch in range(EPOCHS):
    ae.train()
    epoch_loss = 0
    for batch in train_loader:
        xb = batch[0].to(DEVICE)
        optimizer.zero_grad()
        x_hat_raw, x_hat_log, _ = ae(xb)
        loss = criterion(x_hat_raw, xb) + criterion(x_hat_log, torch.log1p(xb))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(X_tr_scaled):.6f}")

# Encode function
def encode_dataset_mv(model, X_data):
    model.eval()
    z_list = []
    with torch.no_grad():
        for i in range(0, len(X_data), BATCH_SIZE):
            batch = torch.from_numpy(X_data[i:i+BATCH_SIZE]).float().to(DEVICE)
            _, _, z = model(batch)
            z_list.append(z.cpu().numpy())
    return np.vstack(z_list)

X_tr_enc = encode_dataset_mv(ae, X_tr_scaled)
X_val_enc = encode_dataset_mv(ae, X_val_scaled)
X_test_enc = encode_dataset_mv(ae, X_test_scaled)

# ------------------ Feature Fusion ------------------
X_tr_comb = np.hstack([X_tr_scaled, X_tr_enc])
X_val_comb = np.hstack([X_val_scaled, X_val_enc])
X_test_comb = np.hstack([X_test_scaled, X_test_enc])

# ------------------ Hybrid Stacked Classifier ------------------
params_lgb = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 128,
    'max_depth': 12,
    'n_estimators': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
params_cat = {'learning_rate': 0.05, 'depth': 10, 'iterations': 1000, 'random_seed': RANDOM_STATE, 'verbose': 0}

estimators = [
    ('lgb', lgb.LGBMClassifier(**params_lgb)),
    ('cat', cb.CatBoostClassifier(**params_cat)),
    ('rf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

print("[INFO] Training hybrid stacked classifier...")
stack_model.fit(X_tr_comb, y_tr)

# ------------------ Threshold tuning ------------------
val_probs = stack_model.predict_proba(X_val_comb)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_f1 = 0.5, 0
for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"[INFO] Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")

# ------------------ Evaluate ------------------
test_probs = stack_model.predict_proba(X_test_comb)[:, 1]
test_pred = (test_probs > best_thresh).astype(int)
acc = accuracy_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_probs)
f1 = f1_score(y_test, test_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------ Save models ------------------
torch.save(ae.state_dict(), os.path.join(MODEL_DIR, "multi_view_ae.pt"))
joblib.dump(stack_model, os.path.join(MODEL_DIR, "stacked_classifier.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(best_thresh, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"\n[DONE] All models + scaler + threshold saved → {MODEL_DIR}")
