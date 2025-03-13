import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import pickle
from io import BytesIO
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class JewelryPredictor:
    def __init__(self, xgboost_model_path, fnn_model_path, xgboost_scaler_path, fnn_scaler_path, pairwise_features_path, face_features_path, earring_features_path, necklace_features_path):
        # Check if all required files exist
        required_files = [xgboost_model_path, fnn_model_path, xgboost_scaler_path, fnn_scaler_path, pairwise_features_path, face_features_path, earring_features_path, necklace_features_path]
        for path in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        # Initialize models and scalers
        print("Loading XGBoost model...")
        self.xgboost_model = xgb.Booster()
        self.xgboost_model.load_model(xgboost_model_path)

        print("Loading FNN model...")
        self.fnn_model = load_model(fnn_model_path)
        self.img_size = (224, 224)
        self.feature_size = 1280  # Expected feature size after averaging
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("Loading XGBoost scaler...")
        with open(xgboost_scaler_path, 'rb') as f:
            self.xgboost_scaler = pickle.load(f)
        if hasattr(self.xgboost_scaler, 'n_features_in_') and self.xgboost_scaler.n_features_in_ != self.feature_size:
            raise ValueError(f"XGBoost scaler feature size mismatch: Expected {self.feature_size}, got {self.xgboost_scaler.n_features_in_}")

        print("Loading FNN scaler...")
        with open(fnn_scaler_path, 'rb') as f:
            self.fnn_scaler = pickle.load(f)
        if hasattr(self.fnn_scaler, 'n_features_in_') and self.fnn_scaler.n_features_in_ != self.feature_size:
            raise ValueError(f"FNN scaler feature size mismatch: Expected {self.feature_size}, got {self.fnn_scaler.n_features_in_}")

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

        # Load reference features for validation
        print("Loading face reference features...")
        self.face_reference_features = np.load(face_features_path, allow_pickle=True).item()

        print("Loading earring reference features...")
        self.earring_reference_features = np.load(earring_features_path, allow_pickle=True).item()

        print("Loading necklace reference features...")
        self.necklace_reference_features = np.load(necklace_features_path, allow_pickle=True).item()

        # Combine jewelry reference features
        self.jewelry_reference_features = {**self.earring_reference_features, **self.necklace_reference_features}

    def extract_features(self, img_data):
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

    def validate_image_type(self, features, reference_features, threshold=0.7):
        """Check if extracted features match the expected category (face or jewelry)."""
        if features is None:
            return False
        features = features.reshape(1, -1)
        similarities = [cosine_similarity(features, ref.reshape(1, -1))[0][0] for ref in reference_features.values()]
        max_similarity = max(similarities) if similarities else 0
        return max_similarity > threshold

    def combine_features(self, face_features, jewelry_features):
        if face_features is None or jewelry_features is None:
            return None
        combined_features = (face_features + jewelry_features) / 2.0  # Shape (1, 1280)
        return combined_features

    def predict_with_xgboost(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        if combined_features is None:
            return None, None
        try:
            combined_features = self.xgboost_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing features with XGBoost scaler: {e}")
            return None, None
        dmatrix = xgb.DMatrix(combined_features)
        score = self.xgboost_model.predict(dmatrix)[0] * 100.0
        score = np.clip(score, 0, 100)
        category = self.compute_category(score)
        return score, category

    def predict_with_fnn(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        if combined_features is None:
            return None, None
        try:
            combined_features = self.fnn_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing features with FNN scaler: {e}")
            return None, None
        with tf.device(self.device):
            score = self.fnn_model.predict(combined_features, verbose=0)[0][0] * 100.0
        score = np.clip(score, 0, 100)
        category = self.compute_category(score)
        return score, category

    def get_recommendations_with_xgboost(self, face_features):
        if not self.jewelry_list:
            print("No pairwise features available for recommendations.")
            return []
        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        combined_features = np.vstack(combined_features)
        try:
            combined_features = self.xgboost_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing jewelry features with XGBoost scaler: {e}")
            return []
        dmatrix = xgb.DMatrix(combined_features)
        scores = self.xgboost_model.predict(dmatrix) * 100.0
        scores = np.clip(scores, 0, 100)
        valid_indices = ~np.isnan(scores)
        scores = scores[valid_indices]
        if len(scores) == 0:
            print("No valid scores after filtering NaN values.")
            return []
        indices = np.argsort(scores)[::-1][:20]
        recommendations = [self.jewelry_names[i] for i in indices]
        rec_scores = scores[indices]
        rec_categories = [self.compute_category(score) for score in rec_scores]
        return [{"name": name, "score": round(float(score), 2), "category": category} for name, score, category in zip(recommendations, rec_scores, rec_categories)]

    def get_recommendations_with_fnn(self, face_features):
        if not self.jewelry_list:
            print("No pairwise features available for recommendations.")
            return []
        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        combined_features = np.vstack(combined_features)
        try:
            combined_features = self.fnn_scaler.transform(combined_features)
        except Exception as e:
            print(f"❌ Error normalizing jewelry features with FNN scaler: {e}")
            return []
        with tf.device(self.device):
            scores = self.fnn_model.predict(combined_features, verbose=0).flatten() * 100.0
        scores = np.clip(scores, 0, 100)
        valid_indices = ~np.isnan(scores)
        scores = scores[valid_indices]
        if len(scores) == 0:
            print("No valid scores after filtering NaN values.")
            return []
        indices = np.argsort(scores)[::-1][:20]
        recommendations = [self.jewelry_names[i] for i in indices]
        rec_scores = scores[indices]
        rec_categories = [self.compute_category(score) for score in rec_scores]
        return [{"name": name, "score": round(float(score), 2), "category": category} for name, score, category in zip(recommendations, rec_scores, rec_categories)]

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

def get_predictor(xgboost_model_path, fnn_model_path, xgboost_scaler_path, fnn_scaler_path, pairwise_features_path, face_features_path, earring_features_path, necklace_features_path):
    try:
        predictor = JewelryPredictor(xgboost_model_path, fnn_model_path, xgboost_scaler_path, fnn_scaler_path, pairwise_features_path, face_features_path, earring_features_path, necklace_features_path)
        return predictor
    except Exception as e:
        print(f"Failed to initialize JewelryPredictor: {e}")
        return None

def predict_both(predictor, face_data, jewelry_data):
    if predictor is None:
        return None, "Predictor not initialized", [], None, "Predictor not initialized", []

    face_features = predictor.extract_features(face_data)
    jewelry_features = predictor.extract_features(jewelry_data)

    if face_features is None or jewelry_features is None:
        return None, "Feature extraction failed", [], None, "Feature extraction failed", []

    # Validate image types
    if not predictor.validate_image_type(face_features, predictor.face_reference_features):
        return None, "Invalid face image", [], None, "Invalid face image", []
    if not predictor.validate_image_type(jewelry_features, predictor.jewelry_reference_features):
        return None, "Invalid jewelry image", [], None, "Invalid jewelry image", []

    xgboost_score, xgboost_category = predictor.predict_with_xgboost(face_features, jewelry_features)
    fnn_score, fnn_category = predictor.predict_with_fnn(face_features, jewelry_features)
    xgboost_recommendations = predictor.get_recommendations_with_xgboost(face_features)
    fnn_recommendations = predictor.get_recommendations_with_fnn(face_features)
    return xgboost_score, xgboost_category, xgboost_recommendations, fnn_score, fnn_category, fnn_recommendations