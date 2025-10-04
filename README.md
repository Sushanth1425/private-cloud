
# Sus-Cloud  
A secure **Private Cloud Storage System** built with the **MERN stack**.  
It allows users to **upload, download, share, and manage files** with authentication and encryption.  

---

##  Features
- **User Authentication (JWT)** – Secure login & registration.  
- **File Upload & Download** – Users can upload and retrieve files.  
- **File Sharing (link-based)** – Generate secure share links.  
- **File Deletion** – Users can delete their files.  
- **Encryption** – Files are encrypted with SM Encryptions before storage.  
- **Clean API structure** with Express.  
- **Ready for frontend integration** (React.js).  

---

##  Project Structure

````
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

---

##  Backend Setup

### 1️. Clone Repository
```bash
git clone https://github.com/Sushanth1425/private-cloud.git
cd private-cloud/backend
````

### 2️. Install Dependencies

```bash
npm install
```

### 3️. Environment Variables

Create a `.env` file inside the `backend/` directory:

```ini
PORT=5050
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_jwt_secret_key
UPLOAD_DIR=./uploads
FRONTEND_URL=your_frontend_url # preferablly (http://localhost:5173/)

```


### 4️. Start Backend Server

```bash
npm start
```

Backend will run at: **[http://localhost:5050](http://localhost:5050)**

---

## 5️. API Endpoints  
**Content:**

## 🛠️ API Endpoints

### Auth Routes (`/api/auth`)
- `POST /register` → Register new user
- `POST /login` → Login user

### File Routes (`/api/files`)
- `POST /upload` → Upload file
- `GET /download/:id` → Download file
- `DELETE /delete/:id` → Delete file
- `POST /share/:id` → Generate shareable link
- `GET /shared/:token` → Access shared file

---

##  API Request Examples

### 1. Register User

**Endpoint:** `POST /api/auth/register`
**Request Body (JSON):**

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123"
}
```

**Response (Success 201):**

```json
{
  "message": "User registered successfully",
  "user": {
    "_id": "64f1a1e2c12345abcdef1234",
    "name": "John Doe",
    "email": "john@example.com"
  },
  "token": "jwt_token_here"
}
```

---

### 2. Login User

**Endpoint:** `POST /api/auth/login`
**Request Body (JSON):**

```json
{
  "email": "john@example.com",
  "password": "password123"
}
```

**Response (Success 200):**

```json
{
  "message": "Login successful",
  "user": {
    "_id": "64f1a1e2c12345abcdef1234",
    "name": "John Doe",
    "email": "john@example.com"
  },
  "token": "jwt_token_here"
}
```

---

### 3. Upload File

**Endpoint:** `POST /api/files/upload`
**Headers:**

```
Authorization: Bearer <jwt_token_here>
Content-Type: multipart/form-data
```

**Form Data:**

```
file: <choose_file>
```

**Response (Success 201):**

```json
{
  "message": "File uploaded successfully",
  "file": {
    "_id": "64f1b2f3c12345abcdef5678",
    "filename": "example.pdf",
    "owner": "64f1a1e2c12345abcdef1234",
    "url": "/uploads/example_encrypted.pdf"
  }
}
```

---

### 4. Download File

**Endpoint:** `GET /api/files/download/:id`
**Headers:**

```
Authorization: Bearer <jwt_token_here>
```

**Response:** Returns the requested file as attachment.

---

### 5. Delete File

**Endpoint:** `DELETE /api/files/delete/:id`
**Headers:**

```
Authorization: Bearer <jwt_token_here>
```

**Response (Success 200):**

```json
{
  "message": "File deleted successfully"
}
```

---

### 6. Generate Shareable Link

**Endpoint:** `POST /api/files/share/:id`
**Headers:**

```
Authorization: Bearer <jwt_token_here>
```

**Response (Success 200):**

```json
{
  "message": "Shareable link generated",
  "link": "http://localhost:5173/shared/abc123token"
}
```

---

### 7. Access Shared File

**Endpoint:** `GET /api/files/shared/:token`
**Response:** Returns the shared file for download.


---




--- 

##  API Error Responses (Examples)

### 1. Register Error
**Response (400 Bad Request):**
```json
{
  "error": "User with this email already exists"
}
````

### 2. Login Error

**Response (401 Unauthorized):**

```json
{
  "error": "Invalid credentials"
}
```

### 3. File Upload Error

**Response (400 Bad Request):**

```json
{
  "error": "No file provided"
}
```

### 4. File Access Error

**Response (403 Forbidden / 404 Not Found):**

```json
{
  "error": "You do not have permission to access this file"
}
```

---

## 🚀 Deployment 

* **Frontend:** Vercel
* **Backend:** Render / AWS EC2
* **Database:** MongoDB Atlas


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).



---




##  Frontend 

* Will be built with **React.js**.
* Integrated with backend APIs using **Axios**.
* Includes **login system, dashboard, file manager UI**.

---

##  Security

* Files are encrypted using custom crypto utils before being stored.
* JWT authentication protects all private routes.
* Sensitive data (`.env`, `keys`, `uploads`) is ignored via `.gitignore`.

---

##  To-Do

* [x] Backend API setup
* [x] Authentication & File Management
* [x] Frontend React.js Dashboard
* [x] File Preview (images, pdfs)
* [ ] Hybrid Malware Detection API 🔬
* [ ] Deploy to Cloud (Vercel + Render/EC2 + MongoDB Atlas)

---

##  Tech Stack

* **Frontend** → React.js 
* **Backend** → Node.js, Express.js
* **Database** → MongoDB
* **Auth** → JWT
* **File Handling** → Multer
* **Encryption** → Custom crypto (SM2 keys, AES, etc.)

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.

---

##  License

MIT License © 2025 [Sushanth B](https://github.com/Sushanth1425)

```

---

```
