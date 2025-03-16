import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import pickle
from io import BytesIO
from PIL import Image
import cv2  # Added OpenCV import

class JewelryPredictor:
    def __init__(self, xgboost_model_path, mlp_model_path, xgboost_scaler_path, mlp_scaler_path, pairwise_features_path):
        # Check if all required files exist
        required_files = [xgboost_model_path, mlp_model_path, xgboost_scaler_path, mlp_scaler_path, pairwise_features_path]
        for path in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        # Initialize models and scalers
        print("Loading XGBoost model...")
        self.xgboost_model = xgb.Booster()
        self.xgboost_model.load_model(xgboost_model_path)

        print("Loading MLP model...")
        self.mlp_model = load_model(mlp_model_path)
        self.img_size = (224, 224)
        self.feature_size = 1280  # Expected feature size after averaging
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("Loading XGBoost scaler...")
        with open(xgboost_scaler_path, 'rb') as f:
            self.xgboost_scaler = pickle.load(f)
        if hasattr(self.xgboost_scaler, 'n_features_in_') and self.xgboost_scaler.n_features_in_ != self.feature_size:
            raise ValueError(f"XGBoost scaler feature size mismatch: Expected {self.feature_size}, got {self.xgboost_scaler.n_features_in_}")

        print("Loading MLP scaler...")
        with open(mlp_scaler_path, 'rb') as f:
            self.mlp_scaler = pickle.load(f)
        if hasattr(self.mlp_scaler, 'n_features_in_') and self.mlp_scaler.n_features_in_ != self.feature_size:
            raise ValueError(f"MLP scaler feature size mismatch: Expected {self.feature_size}, got {self.mlp_scaler.n_features_in_}")

        # Set up feature extractor
        print("Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=GlobalAveragePooling2D()(base_model.output)
        )

        # Load pairwise features
        print("Loading pairwise features...")
        try:
            self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
            self.pairwise_features = {
                k: np.array(v) for k, v in self.pairwise_features.items()
                if v is not None and v.size == self.feature_size and not np.any(np.isnan(v))
            }
        except Exception as e:
            print(f"Error loading pairwise features: {e}")
            self.pairwise_features = {}
        self.jewelry_list = [v.reshape(1, -1) for v in self.pairwise_features.values()]
        self.jewelry_names = list(self.pairwise_features.keys())
        print(f"Loaded {len(self.jewelry_names)} pairwise features successfully!")

        # Initialize OpenCV face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("Error loading Haar cascade classifier")

    def validate_image(self, img_data, is_face=False):
        """Validate if the image is suitable (not blank, correct dimensions, etc.). Use OpenCV for face detection."""
        try:
            # Load image with PIL to check dimensions and format
            img = Image.open(BytesIO(img_data))
            width, height = img.size
            if width < 50 or height < 50:
                print("Image too small")
                return False

            # Convert to array and check format
            img = img.resize(self.img_size)
            img_array = np.array(img)
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                print("Image must be in RGB format")
                return False

            # Check if the image is "empty"
            std_dev = np.std(img_array)
            if std_dev < 10:
                print("Image appears to be too uniform (possibly blank)")
                return False

            # If validating a face image, use OpenCV Haar cascade
            if is_face:
                # Convert PIL image to OpenCV format (BGR)
                cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    print("No faces detected in the image")
                    return False

            return True
        except Exception as e:
            print(f"Error validating image: {e}")
            return False

    def extract_features(self, img_data, is_face=False):
        if not self.validate_image(img_data, is_face=is_face):
            return None
        try:
            img = image.load_img(BytesIO(img_data), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return features  # Returns shape (1, 1280)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def combine_features(self, face_features, jewelry_features):
        if face_features is None or jewelry_features is None:
            return None
        combined_features = (face_features + jewelry_features) / 2.0  # Shape (1, 1280)
        return combined_features

    def predict_with_xgboost(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        if combined_features is None:
            return 50.0, "Neutral"  # Fallback score and category
        try:
            combined_features = self.xgboost_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing features with XGBoost scaler: {e}")
            return 50.0, "Neutral"  # Fallback score and category
        dmatrix = xgb.DMatrix(combined_features)
        score = self.xgboost_model.predict(dmatrix)[0] * 100.0
        score = np.clip(score, 0, 100)
        category = self.compute_category(score)
        return score, category

    def predict_with_mlp(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        if combined_features is None:
            return 50.0, "Neutral"  # Fallback score and category
        try:
            combined_features = self.mlp_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing features with MLP scaler: {e}")
            return 50.0, "Neutral"  # Fallback score and category
        with tf.device(self.device):
            score = self.mlp_model.predict(combined_features, verbose=0)[0][0] * 100.0
        score = np.clip(score, 0, 100)
        category = self.compute_category(score)
        return score, category

    def get_recommendations_with_xgboost(self, face_features):
        # Always return exactly 10 recommendations with a robust fallback
        if not self.jewelry_list or len(self.jewelry_list) == 0:
            print("No jewelry features available, returning fallback recommendations.")
            return [
                {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        if any(cf is None for cf in combined_features):
            print("Invalid combined features detected, returning fallback recommendations.")
            return [
                {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        combined_features = np.vstack(combined_features)
        try:
            combined_features = self.xgboost_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing jewelry features with XGBoost scaler: {e}")
            return [
                {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        dmatrix = xgb.DMatrix(combined_features)
        scores = self.xgboost_model.predict(dmatrix) * 100.0
        scores = np.clip(scores, 0, 100)
        valid_indices = ~np.isnan(scores)
        scores = scores[valid_indices]
        valid_jewelry_names = [self.jewelry_names[i] for i in range(len(self.jewelry_names)) if valid_indices[i]]

        if len(scores) == 0:
            print("No valid scores, returning fallback recommendations.")
            return [
                {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        indices = np.argsort(scores)[::-1]
        recommendations = []
        for i in range(min(10, len(scores))):
            idx = indices[i]
            name = valid_jewelry_names[idx] if idx < len(valid_jewelry_names) else f"fallback_xgboost_{i}"
            score = scores[idx]
            category = self.compute_category(score)
            recommendations.append({"name": name, "score": round(float(score), 2), "category": category})

        # Ensure exactly 10 recommendations with fallbacks if needed
        while len(recommendations) < 10:
            i = len(recommendations)
            recommendations.append({"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"})

        return recommendations

    def get_recommendations_with_mlp(self, face_features):
        # Always return exactly 10 recommendations with a robust fallback
        if not self.jewelry_list or len(self.jewelry_list) == 0:
            print("No jewelry features available, returning fallback recommendations.")
            return [
                {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        if any(cf is None for cf in combined_features):
            print("Invalid combined features detected, returning fallback recommendations.")
            return [
                {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        combined_features = np.vstack(combined_features)
        try:
            combined_features = self.mlp_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing jewelry features with MLP scaler: {e}")
            return [
                {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        with tf.device(self.device):
            scores = self.mlp_model.predict(combined_features, verbose=0).flatten() * 100.0
        scores = np.clip(scores, 0, 100)
        valid_indices = ~np.isnan(scores)
        scores = scores[valid_indices]
        valid_jewelry_names = [self.jewelry_names[i] for i in range(len(self.jewelry_names)) if valid_indices[i]]

        if len(scores) == 0:
            print("No valid scores, returning fallback recommendations.")
            return [
                {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"}
                for i in range(10)
            ]

        indices = np.argsort(scores)[::-1]
        recommendations = []
        for i in range(min(10, len(scores))):
            idx = indices[i]
            name = valid_jewelry_names[idx] if idx < len(valid_jewelry_names) else f"fallback_mlp_{i}"
            score = scores[idx]
            category = self.compute_category(score)
            recommendations.append({"name": name, "score": round(float(score), 2), "category": category})

        # Ensure exactly 10 recommendations with fallbacks if needed
        while len(recommendations) < 10:
            i = len(recommendations)
            recommendations.append({"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"})

        return recommendations

    def compute_category(self, score: float) -> str:
        if score >= 80.0:
            return "Very Good"
        elif score >= 60.0:
            return "Good"
        elif score >= 40.0:
            return "Neutral"
        elif score >= 20.0:
            return "Bad"
        else:
            return "Very Bad"

def get_predictor(xgboost_model_path, mlp_model_path, xgboost_scaler_path, mlp_scaler_path, pairwise_features_path):
    try:
        predictor = JewelryPredictor(xgboost_model_path, mlp_model_path, xgboost_scaler_path, mlp_scaler_path, pairwise_features_path)
        return predictor
    except Exception as e:
        print(f"Failed to initialize JewelryPredictor: {e}")
        return None

def predict_both(predictor, face_data, jewelry_data):
    if predictor is None:
        return 50.0, "Predictor not initialized", [
            {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ], 50.0, "Predictor not initialized", [
            {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ]

    face_features = predictor.extract_features(face_data, is_face=True)
    jewelry_features = predictor.extract_features(jewelry_data, is_face=False)

    if face_features is None:
        return 50.0, "Invalid face image", [
            {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ], 50.0, "Invalid face image", [
            {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ]
    if jewelry_features is None:
        return 50.0, "Invalid jewelry image", [
            {"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ], 50.0, "Invalid jewelry image", [
            {"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"} for i in range(10)
        ]

    xgboost_score, xgboost_category = predictor.predict_with_xgboost(face_features, jewelry_features)
    mlp_score, mlp_category = predictor.predict_with_mlp(face_features, jewelry_features)
    xgboost_recommendations = predictor.get_recommendations_with_xgboost(face_features)
    mlp_recommendations = predictor.get_recommendations_with_mlp(face_features)
    return xgboost_score, xgboost_category, xgboost_recommendations, mlp_score, mlp_category, mlp_recommendations