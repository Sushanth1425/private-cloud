import os
import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# ==================== Config ====================
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
SUBSET_SIZE = 40000  # Adjust if RAM <16GB
TOP_FEATURES = 500

# ==================== Load Dataset ====================
print("[INFO] Loading BODMAS dataset...")
df = pd.read_parquet(os.path.join(DATA_DIR, "bodmas.parquet"))
print(f"[INFO] Dataset loaded: {df.shape}")

# ==================== Auto-detect label column ====================
possible_labels = ["label", "Label", "malicious", "target"]
label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError("No known label column found! Check dataset columns.")

print(f"[INFO] Using '{label_col}' as label column")

# Split features and target
X = df.drop(columns=[label_col])
y = df[label_col].astype(int)
del df
gc.collect()

X = X.select_dtypes(include=[np.number])
print(f"[INFO] Numeric features retained: {X.shape[1]}")


# ==================== Subsample for memory ====================
mal_idx = np.where(y == 1)[0]
ben_idx = np.where(y == 0)[0]

n_mal = min(len(mal_idx), SUBSET_SIZE // 2)
n_ben = min(len(ben_idx), SUBSET_SIZE // 2)

rng = np.random.default_rng(RANDOM_STATE)
mal_sample = rng.choice(mal_idx, n_mal, replace=False)
ben_sample = rng.choice(ben_idx, n_ben, replace=False)

subset_idx = np.concatenate([mal_sample, ben_sample])
rng.shuffle(subset_idx)

X_sub = X.iloc[subset_idx]
y_sub = y.iloc[subset_idx]

print(f"[INFO] Subset ready: {X_sub.shape}, malware ratio: {y_sub.mean():.2f}")

del X, y, mal_idx, ben_idx, subset_idx
gc.collect()

# ==================== Train/Validation/Test split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=RANDOM_STATE
)
del X_sub, y_sub
gc.collect()

# ==================== Scale + Feature Selection ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(mutual_info_classif, k=min(TOP_FEATURES, X_train_scaled.shape[1]))
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

print(f"[INFO] Feature selection done: {X_train_sel.shape}")

# ==================== Train LightGBM ====================
print("[INFO] Training LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lgbm.fit(X_train_sel, y_train)

# ==================== Train Logistic Regression ====================
print("[INFO] Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver='saga',
    n_jobs=-1,
    random_state=RANDOM_STATE
)
lr.fit(X_train_sel, y_train)

# ==================== Hybrid Prediction ====================
print("[INFO] Evaluating hybrid model...")
lgbm_pred_proba = lgbm.predict_proba(X_test_sel)[:, 1]
lr_pred_proba = lr.predict_proba(X_test_sel)[:, 1]

# Hybrid weights (can tune to improve accuracy)
final_proba = 0.7 * lgbm_pred_proba + 0.3 * lr_pred_proba
final_pred = (final_proba > 0.5).astype(int)

acc = accuracy_score(y_test, final_pred)
auc = roc_auc_score(y_test, final_proba)
f1 = f1_score(y_test, final_pred)

print(f"\n✅ [HYBRID MODEL RESULTS]")
print(f"Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, final_pred, zero_division=0))

# ==================== Save Models ====================
print("[INFO] Saving models and transformers...")
joblib.dump(lgbm, os.path.join(MODEL_DIR, "lightgbm_bodmas.pkl"))
joblib.dump(lr, os.path.join(MODEL_DIR, "logreg_bodmas.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_bodmas.pkl"))
joblib.dump(selector, os.path.join(MODEL_DIR, "feature_selector_bodmas.pkl"))

with open(os.path.join(MODEL_DIR, "hybrid_weights.txt"), "w") as f:
    f.write("LightGBM: 0.7\nLogisticRegression: 0.3\n")

print("[DONE] ✅ Hybrid model training complete on BODMAS dataset!")
