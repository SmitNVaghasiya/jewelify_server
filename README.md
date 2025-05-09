# Jewelify

Welcome to **Jewelify**, an AI-powered jewelry recommendation system designed to enhance the jewelry shopping experience using image processing and machine learning. This repository contains both the backend server (built with FastAPI) and the frontend application (built with Flutter), providing a complete solution for personalized jewelry recommendations. Users can upload face and jewelry images, and the system analyzes compatibility, generating a matching score and suggesting up to 10 jewelry options.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Machine Learning Models](#machine-learning-models)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Jewelify leverages image processing and machine learning to provide personalized jewelry recommendations. The system allows users to upload face and jewelry images, analyzes compatibility using AI models, and suggests jewelry options with a matching score. The backend is powered by FastAPI, handling user authentication, predictions, and history storage, while the frontend is built with Flutter, offering a seamless mobile experience.

The application is deployed on **Render** for the backend, with features like JWT-based authentication, image processing for facial recognition, and machine learning model inference for jewelry recommendations.

## Features

- **User Authentication:** Secure sign-up, login, and profile management using JWT.
- **Jewelry Predictions:** Processes face and jewelry images to predict compatibility scores and recommend jewelry.
- **User History:** Stores and retrieves past predictions for user reference.
- **Image Processing:** Detects facial features using OpenCV for AI-based analysis.
- **Scalable API:** Built with FastAPI for high-performance RESTful APIs.
- **Cross-Platform Frontend:** Flutter-based mobile app for a seamless user experience on iOS and Android.
- **Email Notifications:** Uses an SMTP server for sending email-based communications.
- **Deployment:** Backend hosted on Render with a `keep_alive.py` script to maintain service availability.

## Directory Structure

```
.
├── Jewelify_server/              # Backend server code
│   ├── api/                      # API route definitions (auth, predictions, history)
│   ├── models/                   # Pydantic models or database schemas
│   ├── services/                 # Business logic and service integrations
│   ├── trained_features/         # Pre-computed features for ML models
│   ├── .env                      # Environment variables (ensure in .gitignore)
│   ├── .gitignore                # Files to ignore in Git
│   ├── main.py                   # FastAPI application entry point
│   ├── requirements.txt          # Python dependencies
│   ├── keep_alive.py             # Script to keep Render service alive
│   ├── *.model                   # Trained XGBoost model (e.g., xgboost_jewelry_v1.model)
│   ├── *.pkl                     # Pickled scalers (e.g., scaler_xgboost_v1.pkl)
│   ├── *.keras                   # Trained Keras model (e.g., mlp_jewelry_v1.keras)
│   ├── *.npy                     # NumPy arrays (e.g., necklace_features.npy)
│   └── haarcascade_frontalface_default.xml # OpenCV Haar cascade for face detection
├── jewelify_app/                 # Frontend Flutter code (Corrected Directory Name)
│   ├── lib/                      # Dart source files
│   ├── assets/                   # Images, fonts, and other assets
│   ├── pubspec.yaml              # Flutter dependencies and configuration
│   ├── android/                  # Android-specific code
│   ├── ios/                      # iOS-specific code
│   └── test/                     # Unit and widget tests
├── .gitignore                    # Root-level Git ignore file
└── README.md                     # Project documentation
```

## Tech Stack

### Backend

- **FastAPI:** Framework for building RESTful APIs.
- **Uvicorn:** ASGI server to run FastAPI.
- **TensorFlow/Keras & XGBoost:** Machine learning models for jewelry predictions.
- **Scikit-learn:** Utilities for ML (e.g., scalers).
- **OpenCV:** Image processing for face detection.
- **MongoDB:** Database for storing user data and history (inferred from `pymongo` and `motor`).
- **Pydantic:** Data validation for API requests/responses.
- **python-jose & passlib:** JWT authentication and password hashing.
- **SMTP Server:** Email communication for notifications.
- **python-dotenv:** Environment variable management.
- **NumPy:** Handling feature data in `.npy` files.

### Frontend

- **Flutter:** Framework for building cross-platform mobile applications.
- **Dart:** Programming language for Flutter.
- **HTTP Client:** For communicating with the FastAPI backend (e.g., `http` package in Flutter).

## Setup and Installation

### Backend Setup

1. **Navigate to the Backend Directory:**

   ```bash
   cd Jewelify_server
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the `Jewelify_server` directory with the following variables:
   ```
   DATABASE_URL=<your-mongodb-url>
   SECRET_KEY=<your-jwt-secret-key>
   ALGORITHM=HS256
   SMTP_HOST=<your-smtp-host>
   SMTP_PORT=<your-smtp-port>
   SMTP_USER=<your-smtp-user>
   SMTP_PASS=<your-smtp-password>
   PORT=5000
   ```
   Ensure `.env` is added to `.gitignore` to protect sensitive data.

### Frontend Setup

1. **Navigate to the Frontend Directory:**

   ```bash
   cd jewelify_app
   ```

2. **Install Flutter Dependencies:**
   Ensure you have [Flutter](https://flutter.dev/docs/get-started/install) installed. Then run:

   ```bash
   flutter pub get
   ```

3. **Configure API Base URL:**
   Update the API base URL in your Flutter code (e.g., in a constants file) to point to your backend:
   ```dart
   const String apiBaseUrl = 'http://localhost:5000'; // Update with your deployed URL
   ```

## Running the Application

### Running the Backend

1. Ensure dependencies are installed and environment variables are set.
2. Run the FastAPI application:

   ```bash
   cd Jewelify_server
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```

   - `--reload`: Enables auto-reloading for development.
   - `--host` and `--port`: Configurable via the `PORT` environment variable.

3. Access the API at `http://localhost:5000`. Use `/docs` for the Swagger UI to explore endpoints.

### Running the Frontend

1. Ensure Flutter dependencies are installed.
2. Run the Flutter app:
   ```bash
   cd jewelify_app
   flutter run
   ```
   Ensure a device or emulator is connected. The app will connect to the backend API for functionality.

## API Endpoints

| Endpoint           | Method   | Description                                    |
| ------------------ | -------- | ---------------------------------------------- |
| `/`                | GET      | Welcome message                                |
| `/health`          | GET      | Health check for the server                    |
| `/auth/...`        | Multiple | Authentication endpoints (signup, login, etc.) |
| `/predictions/...` | Multiple | Jewelry prediction endpoints                   |
| `/history/...`     | Multiple | User history retrieval                         |

Detailed endpoint definitions are in the `Jewelify_server/api/` directory.

## Machine Learning Models

The application uses pre-trained models for jewelry recommendations:

- **XGBoost Model:** `xgboost_jewelry_v1.model` for compatibility scoring.
- **MLP Model (Keras):** `mlp_jewelry_v1.keras` for neural network predictions.
- **Scalers:** `scaler_xgboost_v1.pkl`, `scaler_mlp_v1.pkl` for data preprocessing.
- **Feature Data:** `.npy` files (e.g., `necklace_features.npy`, `face_features.npy`) for pre-computed features.
- **Face Detection:** `haarcascade_frontalface_default.xml` for OpenCV-based facial feature detection.

These models are integrated into the `/predictions` API to analyze user-uploaded images and provide recommendations.

## Deployment

### Backend Deployment

The backend is deployed on **Render**. Key deployment notes:

- The `keep_alive.py` script pings the `/health` endpoint to prevent the free-tier instance from sleeping.
- Ensure environment variables are configured in Render's dashboard.
- Models, scalers, and feature data must be included in the deployment.

### Frontend Deployment

The Flutter app can be built and deployed as an APK for Android or an IPA for iOS:

- **Android:** `flutter build apk`
- **iOS:** `flutter build ios` (requires a macOS environment and Apple Developer account).

Ensure the API base URL in the Flutter app points to the deployed backend URL.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 guidelines for Python and Flutter best practices for Dart, and include appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if available, or add one).
