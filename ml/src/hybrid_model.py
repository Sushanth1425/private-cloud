import joblib
import numpy as np
import tensorflow as tf

rf = joblib.load("../models/random_forest.pkl")
xgb = joblib.load("../models/xgboost.pkl")
dl_model = tf.keras.models.load_model("../models/deep_model.h5")

def hybrid_predict(features):
    f = np.array(features).reshape(1, -1)
    rf_pred = rf.predict_proba(f)[0][1]
    xgb_pred = xgb.predict_proba(f)[0][1]
    dl_pred = float(dl_model.predict(f, verbose=0)[0][0])

    final_score = (rf_pred + xgb_pred + dl_pred) / 3
    return "Malicious" if final_score > 0.5 else "Benign", final_score
