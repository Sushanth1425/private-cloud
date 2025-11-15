
# SUS-Cloud — Mini Private Cloud with Malware Detection & Secure File Sharing

A Mini Private Cloud built using Raspberry Pi + MERN Stack + FastAPI ML Service designed to offer secure file upload, download, and sharing, with on-device malware detection, AES-based encryption, and private network accessibility via tunneling.

SUS-Cloud ensures that no malicious file ever enters your private cloud storage.

##  Features

### 🔹 1. Secure Private Cloud
- Upload, download, and share files privately
- User Authentication (JWT)
- Encrypted storage on Raspberry Pi HDD

### 🔹 2. AI-Driven Malware Detection
- ML model trained on EMBER malware dataset
- FastAPI inference server
- Extracts features + predicts using ML ensemble
- Blocks malicious files before they reach disk

### 🔹 3. Full MERN Web App
- React Frontend
- Node.js Express Backend
- MongoDB for users, metadata, logs
- Real-time malicious logs + alerts

### 🔹 4. Hosting & Tunneling
- **Frontend**: Vercel 
- **Backend**: Raspberry Pi
- **ML API**:  Render 

### 🔹 5. Compatibility
- Works on LAN, hotspot, or remote access
- Cross-platform Web UI

##  Project Architecture

```plaintext
sus-cloud/
│
├── backend/              # Express.js Backend
│   ├── middlewares/      # Authentication & other middlewares
│   │   └── authMiddleware.js
│   ├── models/           # MongoDB Schemas (User, File)
│   │   ├── User.js
│   │   └── File.js
│   ├── routes/           # API Routes (auth, files)
│   │   ├── auth.js
│   │   └── files.js
│   ├── utils/            # Crypto utils, DB connection
│   │   └── crypto.js
│   ├── server.js         # Entry point for backend
│   └── .gitignore        # Ignores sensitive backend files
│
├── frontend/             # React.js Frontend 
│   ├── .gitignore        # Ignores sensitive frontend files
│   ├── .env              # Frontend environment variables
│   ├── package.json      # Frontend dependencies and scripts
│   ├── index.html        # HTML entry point for frontend
│   ├── vite.config.js    # Vite configuration
│   ├── src/              # Source files
│   │   ├── api/          # API calls
│   │   │   └── api.js
│   │   ├── App.jsx       # Main app component
│   │   ├── index.css     # Global styles
│   │   ├── main.jsx      # Entry point for React app
│   │   ├── routes/       # Route components
│   │   │   └── PrivateRoute.jsx
│   │   ├── pages/        # Pages for routing
│   │   │   ├── Login.jsx
│   │   │   ├── Register.jsx
│   │   │   └── Dashboard.jsx
│   │   └── components/   # Reusable UI components
│   │       ├── Navbar.jsx
│   │       ├── FileUpload.jsx
│   │       ├── FileList.jsx
│   │       └── SharePopup.jsx
│   └── .gitignore        # Ignores sensitive frontend files
│
├── ml/                   # Machine Learning folder
│   ├── data/             # Dataset files (ignored in Git)
│   ├── models/           # Trained models (ignored in Git)
│   ├── src/              # ML scripts and notebooks
│   ├── venv/             # Virtual environment (ignored in Git)
│   └── .gitignore        # Ignores data, models, venv, caches
│
├── .gitignore            # Root ignore file (ignores node_modules, .env, etc.)
├── README.md             # Project documentation

````

##  Machine Learning (EMBER Dataset)

### Models Used

* LightGBM
* Random Forest
* XGBoost
* CatBoost
* Logistic Regression Meta-Learner

### Pipeline

1. Upload file
2. Features extracted (PE Header features)
3. Model inference

#### Response:

```json
{
  "label": "malicious",
  "confidence": 0.9821
}
```

* **If malicious**: file is rejected
* **If benign**: Node backend stores file in HDD

##  FastAPI — Malware Detection API

### Run Locally

```bash
cd ml-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

* **POST /scan**
* **POST** `http://<raspberry-ip>:8000/scan`

#### Body (multipart/form-data):

Returns:

```json
{
  "file_name": "test.exe",
  "prediction": "benign",
  "probability": 0.0432
}
```

##  Node.js Backend (Express)

### Start Server

```bash
npm install
npm start
```

### Responsibilities

* Receives file from user
* Sends file to FastAPI `/scan`
* Allows upload only if safe
* Stores file inside Pi HDD
* JWT auth
* Logs every action in MongoDB

##  React Frontend

Built using:

* React + React Router
* Axios
* Context API
*  Tailwind CSS

### Pages:

* Login 
* Register
* Dashboard




### Running Frontend

```bash
npm install
npm run dev
```

##  MongoDB

### Collections:

* users
* files


Used for:

* File metadata
* Login sessions
* Malware logs
* Sharing links

##  Encryption Used

* **SM2 Encryption**: File encrypted before writing to HDD
* Key stored securely in environment variables
* **SM 3 and SM 4** mode for integrity
* **Password Hashing**: bcrypt
* **Tokens**: JWT (access + refresh)

##  Hosting / Deployment Explained

### ✔ Frontend (React)

* **Vercel** 


### ✔ Backend (Node.js)

* Runs on Raspberry Pi
* Accessible inside LAN or hotspot
* Can expose using:
  * Cloudflare Tunnel 


### ✔ ML API (FastAPI)

* Options:

  * Run on same Raspberry Pi 
  * Host separately on Render


### Why Raspberry Pi?

* Private cloud stays local
* No third-party server dependency
* Secure + cost-effective

##  Novelty 

### Compared to Google Drive / Dropbox / OneDrive:

| Feature                      | Cloud Services | SUS-Cloud                    |
| ---------------------------- | -------------- | ---------------------------- |
| Malware scanning             | Basic          | ML trained on EMBER dataset |
| Private local cloud          | ❌              | ✔                            |
| Encryption before disk write | ❌              | ✔ (SM2/SM3/SM4)              |
| Self-hosted on Raspberry Pi  | ❌              | ✔                            |
| Custom ML API                | ❌              | ✔                            |
| No recurring charges         | ❌              | ✔                            |
| Works offline                | ❌              | ✔                            |

---

```
