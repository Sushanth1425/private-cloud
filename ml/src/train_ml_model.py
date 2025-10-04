import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os
import gc

DATA_DIR = "../data/"

# ============ Step 1: Load Data ============
def load_ember_data():
    print("[INFO] Loading Parquet feature files...")
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "test_ember_2018_v2_features.parquet"))

    print("[INFO] Loading labels...")
    y_train = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)
    y_test = np.fromfile(os.path.join(DATA_DIR, "y_test.dat"), dtype=np.uint8)

    print("[INFO] Aligning label lengths...")
    y_train = y_train[:len(X_train)]
    y_test = y_test[:len(X_test)]

    print("[INFO] Normalizing labels (binary)...")
    y_train = (y_train > 127).astype(int)
    y_test = (y_test > 127).astype(int)

    print("[INFO] Converting to numpy arrays (float32)...")
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    return X_train, X_test, y_train, y_test


# ============ Step 2: Memory-Safe Sampling ============
print("[INFO] Loading data...")
X_train, X_test, y_train, y_test = load_ember_data()
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Instead of full undersampling, we’ll manually select a smaller balanced subset
subset_size = 50000  # Fits 16 GB safely
print(f"[INFO] Selecting {subset_size} balanced samples...")

# Find indices for each class
class0_idx = np.where(y_train == 0)[0]
class1_idx = np.where(y_train == 1)[0]

# Compute proportional samples
num_class1 = min(len(class1_idx), subset_size // 2)
num_class0 = subset_size - num_class1

np.random.seed(42)
sample_idx0 = np.random.choice(class0_idx, num_class0, replace=False)
sample_idx1 = np.random.choice(class1_idx, num_class1, replace=False)

# Combine and shuffle
subset_idx = np.concatenate([sample_idx0, sample_idx1])
np.random.shuffle(subset_idx)

# Extract samples
X_train_sub = X_train[subset_idx]
y_train_sub = y_train[subset_idx]

print(f"[INFO] Subset shape: {X_train_sub.shape}, Malware ratio: {np.mean(y_train_sub):.2f}")

# Free unused memory
del X_train, y_train
gc.collect()


# ============ Step 3: Train Models ============
print("[INFO] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=60,
    max_depth=10,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42
)
rf.fit(X_train_sub, y_train_sub)

print("[INFO] Training XGBoost...")
scale_pos_weight = int(len(y_train_sub[y_train_sub == 0]) / len(y_train_sub[y_train_sub == 1]))
xgb = XGBClassifier(
    eval_metric='logloss',
    n_estimators=80,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train_sub, y_train_sub)


# ============ Step 4: Evaluate ============
print("[INFO] Evaluating models...")

rf_pred_proba = rf.predict_proba(X_test)[:, 1]
xgb_pred_proba = xgb.predict_proba(X_test)[:, 1]

rf_pred = (rf_pred_proba > 0.5).astype(int)
xgb_pred = (xgb_pred_proba > 0.5).astype(int)

print("\n[Random Forest Performance]")
print(classification_report(y_test, rf_pred, zero_division=0))

print("\n[XGBoost Performance]")
print(classification_report(y_test, xgb_pred, zero_division=0))


# ============ Step 5: Hybrid Ensemble ============
print("[INFO] Evaluating Hybrid Model (RF + XGB)...")
final_proba = (rf_pred_proba * 0.4 + xgb_pred_proba * 0.6)
final_pred = (final_proba > 0.5).astype(int)

print("\n[Hybrid Model Performance]")
print(classification_report(y_test, final_pred, zero_division=0))


# ============ Step 6: Save Models ============
print("[INFO] Saving models...")
os.makedirs("../models", exist_ok=True)
joblib.dump(rf, "../models/random_forest.pkl")
joblib.dump(xgb, "../models/xgboost.pkl")

with open("../models/hybrid_weights.txt", "w") as f:
    f.write("RandomForest: 0.4\nXGBoost: 0.6\n")

print("[DONE] Training complete ✅")
