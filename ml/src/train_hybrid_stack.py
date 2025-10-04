import os
import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# ------------------ Config ------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
SUBSET_SIZE = 30000       # 15k malware + 15k benign
K_FEATURES = 300          # reduce 2382 -> 300 features
RANDOM_STATE = 42
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ Load EMBER data ------------------
def load_ember_data():
    print("[INFO] Loading EMBER features...")
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "test_ember_2018_v2_features.parquet"))

    print("[INFO] Loading labels...")
    y_train = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)[:len(X_train)]
    y_test = np.fromfile(os.path.join(DATA_DIR, "y_test.dat"), dtype=np.uint8)[:len(X_test)]

    y_train = (y_train > 127).astype(int)
    y_test = (y_test > 127).astype(int)

    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), y_train, y_test

X_train_full, X_test_full, y_train_full, y_test = load_ember_data()
print(f"[INFO] Full train shape: {X_train_full.shape}, test shape: {X_test_full.shape}")

# ------------------ Balanced subset sampling ------------------
mal_idx = np.where(y_train_full == 1)[0]
ben_idx = np.where(y_train_full == 0)[0]
n_mal = min(len(mal_idx), SUBSET_SIZE // 2)
n_ben = SUBSET_SIZE - n_mal
rng = np.random.RandomState(RANDOM_STATE)
mal_sample = rng.choice(mal_idx, n_mal, replace=False)
ben_sample = rng.choice(ben_idx, n_ben, replace=False)
subset_idx = np.concatenate([mal_sample, ben_sample])
rng.shuffle(subset_idx)

X_sub = X_train_full[subset_idx]
y_sub = y_train_full[subset_idx]

# Free memory
del X_train_full, y_train_full
gc.collect()
print(f"[INFO] Subset ready: {X_sub.shape}, malware ratio: {y_sub.mean():.2f}")

# ------------------ Train/Validation split ------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=RANDOM_STATE
)
del X_sub, y_sub
gc.collect()

# ------------------ Scaling & Feature Selection ------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_full)

selector = SelectKBest(mutual_info_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_val_sel = selector.transform(X_val_scaled)
X_test_sel = selector.transform(X_test_scaled)

# Free memory
del X_train_scaled, X_val_scaled, X_train, X_val, X_test_scaled
gc.collect()
print(f"[INFO] Feature-selected shapes: train {X_train_sel.shape}, val {X_val_sel.shape}, test {X_test_sel.shape}")

# ------------------ Define models ------------------
rf = RandomForestClassifier(
    n_estimators=80,
    max_depth=20,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

lgb = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# ------------------ Voting Ensemble ------------------
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lgb', lgb)],
    voting='soft',
    weights=[1, 2]  # give LGB more influence
)

print("[INFO] Training ensemble...")
ensemble.fit(X_train_sel, y_train)

# ------------------ Evaluate ------------------
print("[INFO] Validation performance:")
val_pred = ensemble.predict(X_val_sel)
val_acc = accuracy_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, ensemble.predict_proba(X_val_sel)[:, 1])
print(f"Validation Accuracy: {val_acc*100:.2f}%, ROC-AUC: {val_auc:.4f}")

print("\n[INFO] Test performance:")
test_pred = ensemble.predict(X_test_sel)
test_acc = accuracy_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_sel)[:, 1])
print(f"Test Accuracy: {test_acc*100:.2f}%, ROC-AUC: {test_auc:.4f}")

print("\nClassification Report (Test):")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------ Save models ------------------
joblib.dump(ensemble, os.path.join(MODEL_DIR, "hybrid_rf_lgb.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(selector, os.path.join(MODEL_DIR, "feature_selector.pkl"))
print(f"[DONE] Ensemble + artifacts saved to {MODEL_DIR}")
