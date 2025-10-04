import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os

DATA_DIR = "../data/ember2018/"

def load_ember_data():
    print("[INFO] Loading Parquet feature files...")
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "train_ember_2018_v2_features.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "test_ember_2018_v2_features.parquet"))

    print("[INFO] Loading labels...")
    y_train = np.fromfile(os.path.join(DATA_DIR, "y_train.dat"), dtype=np.uint8)
    y_test = np.fromfile(os.path.join(DATA_DIR, "y_test.dat"), dtype=np.uint8)

    print("[INFO] Converting to numpy arrays...")
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    return X_train, X_test, y_train, y_test


print("[INFO] Loading data...")
X, y = load_ember_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("[INFO] Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
model.save("../models/deep_model.h5")

print("[DONE] Deep learning model saved!")
