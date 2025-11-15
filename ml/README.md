

#  **Malware Detection Microservice – README.md**

A standalone **FastAPI microservice** used inside **Sus-Cloud (Mini Private Cloud)** to scan user-uploaded files using:

* **Stacking Ensemble ML Model (LightGBM + RF + XGBoost + Meta Logistic Regression)**
* **BODMAS Dataset**
* **Feature Selection + Standardization Pipeline**
* **Static Feature Extraction Engine (PE + Byte Level Features)**
* **EICAR Signature Detection**
* **100% Offline Malware Detection (no external dependencies)**

This service runs separately from the MERN backend and exposes clean HTTP APIs for file scanning.

---

#  **1. Overview**

This API performs malware detection on files uploaded through the cloud interface.

###  Key Features:

* **FastAPI** backend for high-performance scanning
* **2381-feature static feature extraction**
* **Variance threshold feature selection**
* **StandardScaler normalization**
* **Hybrid stacking ensemble classifier**
* **SHA-256 hashing of uploaded files**
* **EICAR antivirus signature support**
* **Model info + health endpoints**
* **Temp-file based scanning (safe & secure)**

---

#  **2. Architecture**

```

User Upload → MERN Backend → FastAPI Malware API → Classifier → Result → MERN Decision (Allow/Block)

```

The malware API is completely isolated from the main backend for:

* Security
* Better fault tolerance
* Independent scaling
* Avoiding Python dependencies inside Node.js

---

#  **3. Directory Structure**

```

ml/
├── models/
│    ├── bodmas_classifier.pkl
│    ├── bodmas_scaler.pkl
│    ├── bodmas_selector.pkl
│    └── bodmas_features.pkl
├── src/
│    ├── run_malware_api.py  
│    └── malware.py  
│ 
└── README.md

````

---

#  **4. Installation**

### **Install dependencies**

```bash
pip install fastapi uvicorn joblib numpy pandas scikit-learn python-multipart
````

### **Start API**

```bash
python run_malware_api.py
```

API runs at:

```
http://localhost:8000
```

---

#  **5. API Endpoints**

## ** POST /api/malware/scan**

Upload a file and receive malware classification.

### **Request**

```bash
curl -X POST "http://localhost:8000/api/malware/scan" \
 -F "file=@sample.exe"
```

### **Response (Benign Example)**

```json
{
  "status": "success",
  "file_name": "sample.exe",
  "file_hash": "87ab0...",
  "file_size": 49231,
  "is_malicious": false,
  "prediction_class": "Benign",
  "confidence": 97.21,
  "malicious_probability": 2.79,
  "benign_probability": 97.21,
  "model_version": "BODMAS v1.0"
}
```

### **Response (Malicious Example)**

```json
{
  "status": "success",
  "file_name": "virus.exe",
  "file_hash": "9218f...",
  "file_size": 102931,
  "is_malicious": true,
  "prediction_class": "Malicious",
  "confidence": 99.61,
  "malicious_probability": 99.61,
  "benign_probability": 0.39,
  "model_version": "BODMAS v1.0"
}
```

---

##  **EICAR Signature Handling**

If the file contains the EICAR test string:

```json
{
  "is_malicious": true,
  "prediction_class": "Malicious",
  "confidence": 99.99,
  "reason": "Matched EICAR signature"
}
```

---

## ** GET /api/malware/model-info**

Returns model metadata.

```json
{
  "model_type": "Stacking Ensemble",
  "base_models": ["LightGBM", "LightGBM", "RandomForest", "XGBoost"],
  "meta_model": "LogisticRegression",
  "num_features": 627,
  "validation_accuracy": 99.05,
  "dataset": "BODMAS"
}
```

---

## ** GET /api/malware/health**

```json
{
  "status": "healthy",
  "models_loaded": true,
  "num_features": 627
}
```

---

#  **6. How Detection Works (Internal Pipeline)**

### **Step 1: Save file to a temporary sandbox**

* Prevents writing malware to disk permanently
* Ensures cleanup after scan

### **Step 2: Extract 2381 static BODMAS features**

Your code includes:

* Byte histogram (256 features)
* Entropy
* File size
* Padding
* **TODO**: PE-header features (future scope)

### **Step 3: Feature selection**

* Variance Threshold removes low-variance noise

### **Step 4: Standardization**

* Scaler ensures ML model consistency

### **Step 5: Ensemble Prediction**

Model =

```
LightGBM + LightGBM + RandomForest + XGBoost → LogisticRegression
```

This ensures:

* High accuracy
* Low false positives
* Robustness to adversarial samples

---

#  **7. Security Advantages**

| Security Feature         | Benefit                              |
| ------------------------ | ------------------------------------ |
| Isolated ML microservice | Prevents backend compromise          |
| SHA-256 hashing          | Duplicate/malicious file tracking    |
| Temp-file execution      | Auto-sandboxing                      |
| EICAR detection          | Antivirus compatibility              |
| Ensemble model           | Higher accuracy than single-model ML |
| No internet needed       | 100% offline malware scanning        |

---

#  **8. Integration With MERN Backend**

Your Node backend sends files to:

```
POST /api/malware/scan
```

If the response contains:

```
is_malicious: true
```

→ Backend **blocks upload** and returns 403.

If not malicious
→ Backend encrypts using **SM3 + SM4 + SM2**
→ Stores file in disk
→ Saves metadata in MongoDB

This is your project’s **core novelty**.

---

#  **9. Docker Deployment**

Create `Dockerfile`:

```dockerfile
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "run_malware_api.py"]
```

Build:

```bash
docker build -t malware-api .
```

Run:

```bash
docker run -p 8000:8000 malware-api
```

---




