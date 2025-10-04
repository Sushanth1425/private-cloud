import os, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib

# ------------------ Config ------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_FEATURES = 300      # Reduce memory usage
MAX_SAMPLES = 100_000   # Subset for 16GB RAM
VAL_SPLIT = 0.1

# ------------------ Load EMBER data ------------------
print("[INFO] Loading EMBER data...")
X = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
y = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)[:len(X)]
y = (y > 127).astype(np.uint8)
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

X = X.iloc[sample_idx].copy().astype(np.float32)  # convert to float32
y = y[sample_idx]

del pos_idx, neg_idx, sample_idx
gc.collect()

# ------------------ Feature selection (variance-based) ------------------
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
X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ------------------ Train CatBoost ------------------
print("[INFO] Training CatBoost...")
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=10,
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    auto_class_weights='Balanced',
    early_stopping_rounds=50,
    verbose=100,
    task_type='CPU',       # CPU-friendly
    thread_count=4
)

train_pool = Pool(X_tr, y_tr)
val_pool = Pool(X_val, y_val)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# ------------------ Threshold tuning ------------------
val_probs = model.predict_proba(X_val)[:,1]
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_f1 = 0.5, 0

for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"[INFO] Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")

# ------------------ Evaluate ------------------
test_probs = model.predict_proba(X_test)[:,1]
test_pred = (test_probs > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_probs)
f1 = f1_score(y_test, test_pred)

print(f"\n✅ Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------ Save model, scaler, threshold ------------------
joblib.dump(model, os.path.join(MODEL_DIR, "catboost_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(best_thresh, os.path.join(MODEL_DIR, "threshold.pkl"))
print(f"\n[DONE] Model + scaler + threshold saved → {MODEL_DIR}")
