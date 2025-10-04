import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
import gc

DATA_DIR = "../data/"

# ==================== Step 1: Load Data ====================
def load_ember_data():
    print("[INFO] Loading Parquet feature files...")
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "test_ember_2018_v2_features.parquet"))

    print("[INFO] Loading labels...")
    y_train = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)
    y_test = np.fromfile(os.path.join(DATA_DIR, "y_test.dat"), dtype=np.uint8)

    y_train = y_train[:len(X_train)]
    y_test = y_test[:len(X_test)]

    y_train = (y_train > 127).astype(int)
    y_test = (y_test > 127).astype(int)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    return X_train, X_test, y_train, y_test


# ==================== Step 2: Sample to Fit RAM ====================
print("[INFO] Loading data...")
X_train, X_test, y_train, y_test = load_ember_data()
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

subset_size = 40000  # works fine on 16GB RAM
print(f"[INFO] Sampling {subset_size} balanced records...")

mal_idx = np.where(y_train == 1)[0]
ben_idx = np.where(y_train == 0)[0]

n_mal = subset_size // 2
n_ben = subset_size - n_mal

np.random.seed(42)
mal_sample = np.random.choice(mal_idx, n_mal, replace=False)
ben_sample = np.random.choice(ben_idx, n_ben, replace=False)

subset_idx = np.concatenate([mal_sample, ben_sample])
np.random.shuffle(subset_idx)

X_train_sub = X_train[subset_idx]
y_train_sub = y_train[subset_idx]

del X_train, y_train
gc.collect()

print(f"[INFO] Subset ready: {X_train_sub.shape}, Malware ratio: {np.mean(y_train_sub):.2f}")

# ==================== Step 3: Feature Scaling + Selection ====================
print("[INFO] Scaling and selecting top features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(mutual_info_classif, k=500)
X_train_sel = selector.fit_transform(X_train_scaled, y_train_sub)
X_test_sel = selector.transform(X_test_scaled)

del X_train_sub, X_train_scaled
gc.collect()

print(f"[INFO] Feature selection done: {X_train_sel.shape}")

# ==================== Step 4: Train Models ====================
print("[INFO] Training LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_train_sel, y_train_sub)

print("[INFO] Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver='saga',
    n_jobs=-1,
    random_state=42
)
lr.fit(X_train_sel, y_train_sub)

# ==================== Step 5: Hybrid Prediction ====================
print("[INFO] Evaluating hybrid model...")
lgbm_pred_proba = lgbm.predict_proba(X_test_sel)[:, 1]
lr_pred_proba = lr.predict_proba(X_test_sel)[:, 1]

final_proba = (0.7 * lgbm_pred_proba + 0.3 * lr_pred_proba)
final_pred = (final_proba > 0.5).astype(int)

acc = accuracy_score(y_test, final_pred)
print(f"\n✅ [HYBRID MODEL ACCURACY]: {acc*100:.2f}%\n")
print(classification_report(y_test, final_pred, zero_division=0))

# ==================== Step 6: Save Everything ====================
print("[INFO] Saving models and transformers...")
os.makedirs("../models", exist_ok=True)
joblib.dump(lgbm, "../models/lightgbm_model.pkl")
joblib.dump(lr, "../models/logistic_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(selector, "../models/feature_selector.pkl")

with open("../models/hybrid_weights.txt", "w") as f:
    f.write("LightGBM: 0.7\nLogisticRegression: 0.3\n")

print("[DONE] Model training complete 🎯 (v2 Hybrid)")
