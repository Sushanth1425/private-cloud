import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib

# ------------------ Config ------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_SAMPLES = 80_000  # ~150k malware + 150k benign
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

X = X.iloc[sample_idx].copy()
y = y[sample_idx]

del pos_idx, neg_idx, sample_idx
gc.collect()

# ------------------ Split Train/Val/Test ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SPLIT, stratify=y_train, random_state=RANDOM_STATE
)
del X_train, y_train, X, y
gc.collect()

# ------------------ Separate numeric & binary features ------------------
num_cols = X_tr.select_dtypes(include=['float32','float64']).columns
bin_cols = X_tr.select_dtypes(include=['int','uint8','bool']).columns

scaler = StandardScaler()
X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ------------------ LightGBM Dataset ------------------
dtrain = lgb.Dataset(X_tr.values, label=y_tr)
dval = lgb.Dataset(X_val.values, label=y_val, reference=dtrain)


# ------------------ Train LightGBM ------------------
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 256,
    'max_depth': 16,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'is_unbalance': True,
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 1e-3,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': RANDOM_STATE
}
num_boost_round = 500

# Early stopping using callbacks
early_stopping = lgb.early_stopping(50)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),  # Early stopping
    lgb.log_evaluation(period=50)  # Logging every 50 rounds
]


print("[INFO] Training LightGBM...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dval],
    #valid_sets=[dtrain, dval],
    #valid_names=['train','val'],
    callbacks=callbacks # Use the callback here
)

# ------------------ Threshold tuning ------------------
val_probs = model.predict(X_val, num_iteration=model.best_iteration)
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_f1 = 0.5, 0

for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"[INFO] Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")

# ------------------ Evaluate ------------------
test_probs = model.predict(X_test, num_iteration=model.best_iteration)
test_pred = (test_probs > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_probs)
f1 = f1_score(y_test, test_pred)

print(f"\n✅ Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------ Save ------------------
joblib.dump(model, os.path.join(MODEL_DIR, "lightgbm_model_highacc.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_highacc.pkl"))
joblib.dump(best_thresh, os.path.join(MODEL_DIR, "threshold_highacc.pkl"))
print(f"\n[DONE] Model + scaler + threshold saved → {MODEL_DIR}")
