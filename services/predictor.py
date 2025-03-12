import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
from io import BytesIO

class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        for path in [model_path, scaler_path, pairwise_features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        print("🚀 Loading model...")
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        self.feature_size = 1280
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("📏 Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("🔄 Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        reduction_layer = tf.keras.layers.Dense(self.feature_size, activation="relu")
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=reduction_layer(global_avg_layer(base_model.output))
        )

        print("📂 Loading pairwise features...")
        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.pairwise_features = {
            k: self.scaler.transform(np.array(v).reshape(1, -1))
            for k, v in self.pairwise_features.items() if v is not None and v.size == self.feature_size
        }
        self.jewelry_list = list(self.pairwise_features.values())
        self.jewelry_names = list(self.pairwise_features.keys())
        print("✅ Predictor initialized successfully!")

    def extract_features(self, img_data):
        try:
            img = image.load_img(BytesIO(img_data), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None

    def predict_compatibility(self, face_data, jewel_data):
        face_features = self.extract_features(face_data)
        jewel_features = self.extract_features(jewel_data)
        if face_features is None or jewel_features is None:
            return None, "Feature extraction failed", []

        face_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        jewel_norm = jewel_features / np.linalg.norm(jewel_features, axis=1, keepdims=True)
        cosine_similarity = np.sum(face_norm * jewel_norm, axis=1)[0]
        # Normalize scaled_score to 0-100% range
        scaled_score = min(max((cosine_similarity + 1) / 2.0, 0.0), 1.0) * 100.0

        if scaled_score >= 80.0:
            category = "Very Good"
        elif scaled_score >= 60.0:
            category = "Good"
        elif scaled_score >= 40.0:
            category = "Neutral"
        elif scaled_score >= 20.0:
            category = "Bad"
        else:
            category = "Very Bad"

        print(f"Category: {category}, Normalized Score: {scaled_score:.2f}%")

        # Compute compatibility for each recommendation using cosine similarity
        recommendations = []
        face_features_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        for idx, (jewel_name, jewel_feature) in enumerate(zip(self.jewelry_names, self.jewelry_list)):
            jewel_features_norm = jewel_feature / np.linalg.norm(jewel_feature, axis=1, keepdims=True)
            rec_cosine_similarity = np.sum(face_features_norm * jewel_features_norm, axis=1)[0]
            rec_score = min(max((rec_cosine_similarity + 1) / 2.0, 0.0), 1.0) * 100.0
            rec_category = self.compute_category(rec_score)
            # Only include recommendations with a score >= 60 (Good or better)
            if rec_score >= 60.0:
                recommendations.append({
                    "name": jewel_name,
                    "url": None,  # URL will be filled by database lookup
                    "score": round(rec_score, 2),  # Round to 2 decimal places
                    "category": rec_category
                })

        # Sort recommendations by score in descending order
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        # Limit to top 10, but only if they meet the threshold
        recommendations = recommendations[:10]

        print(f"Found {len(recommendations)} recommendations with compatibility score >= 60%")
        if len(recommendations) == 0:
            print("No recommendations meet the minimum compatibility threshold of 60%")

        return round(scaled_score, 2), category, recommendations

    def compute_category(self, score: float) -> str:
        """Helper function to compute category based on score (0-100%)."""
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

def get_predictor(model_path, scaler_path, pairwise_features_path):
    try:
        predictor = JewelryRLPredictor(model_path, scaler_path, pairwise_features_path)
        return predictor
    except Exception as e:
        print(f"🚨 Failed to initialize JewelryRLPredictor: {e}")
        return None

def predict_compatibility(predictor, face_data, jewelry_data):
    if predictor is None:
        return None, "Predictor not initialized", []
    return predictor.predict_compatibility(face_data, jewelry_data)