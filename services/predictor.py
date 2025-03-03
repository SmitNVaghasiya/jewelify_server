import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
from io import BytesIO
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging

logger = logging.getLogger(__name__)


class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        """Initialize the predictor with model, scaler, and pairwise features."""
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
        """Extract features from an image."""
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
        """Predict compatibility between a face and jewelry."""
        face_features = self.extract_features(face_data)
        jewel_features = self.extract_features(jewel_data)
        if face_features is None or jewel_features is None:
            return None, "Feature extraction failed", []

        # Normalize features
        face_norm = face_features / np.linalg.norm(face_features, axis=1, keepdims=True)
        jewel_norm = jewel_features / np.linalg.norm(jewel_features, axis=1, keepdims=True)
        cosine_similarity = np.sum(face_norm * jewel_norm, axis=1)[0]
        scaled_score = (cosine_similarity + 1) / 2.0

        # Assign categories based on score with emoji codes
        if scaled_score >= 0.8:
            category = "\u2728 Very Good"  # 🌟 (U+2728)
        elif scaled_score >= 0.6:
            category = "\u2705 Good"  # ✅ (U+2705)
        elif scaled_score >= 0.4:
            category = "\u2639 Neutral"  # 😐 (U+2639, simpler neutral face)
        elif scaled_score >= 0.2:
            category = "\u26A0 Bad"  # ⚠️ (U+26A0)
        else:
            category = "\u274C Very Bad"  # ❌ (U+274C)

        # if scaled_score >= 0.8:
        #     category = "Very Good"  # 🌟 (U+2728)
        # elif scaled_score >= 0.6:
        #     category = "Good"  # ✅ (U+2705)
        # elif scaled_score >= 0.4:
        #     category = "Neutral"  # 😐 (U+2639, simpler neutral face)
        # elif scaled_score >= 0.2:
        #     category = "Bad"  # ⚠️ (U+26A0)
        # else:
        #     category = "Very Bad"  # ❌ (U+274C)

        print(f"Category with emoji code: {category}")  # Debug print to verify

        # Get recommendations from RL model
        with tf.device(self.device):
            q_values = self.model.predict(face_features, verbose=0)[0]
        
        # Check to ensure that the number of Q-values matches the jewelry list length
        if len(q_values) != len(self.jewelry_names):
            print("❌ Error: The number of Q-values does not match the number of jewelry items.")
            return scaled_score, category, []
        
        top_indices = np.argsort(q_values)[::-1]
        top_recommendations = [(self.jewelry_names[idx], q_values[idx]) for idx in top_indices[:10]]
        recommendations = [name for name, _ in top_recommendations]

        return scaled_score, category, recommendations
    
def get_db_client():
    # Your MongoDB client setup logic here
    return MongoClient("mongodb://localhost:27017/")  # Adjust as needed

def get_all_predictions(user_id):
    client = get_db_client()
    if not client:
        logger.error("Failed to connect to MongoDB")
        return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        predictions = list(predictions_collection.find({"user_id": ObjectId(user_id)}).sort("timestamp", -1))
        if not predictions:
            logger.warning(f"No predictions found for user {user_id}")
            return {"error": "No predictions found"}

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for name in recommendations:
                image_doc = images_collection.find_one({"name": name})
                image_data.append({
                    "name": name,
                    "url": image_doc["url"] if image_doc else None
                })

            results.append({
                "id": str(prediction["_id"]),
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "timestamp": prediction["timestamp"]
            })

        logger.info(f"Retrieved {len(results)} predictions for user {user_id}")
        return results
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        return {"error": str(e)}

def predict_compatibility(predictor, face_data, jewelry_data):
    """Wrapper for predict_compatibility method."""
    if predictor is None:
        return None, "Predictor not initialized", []
    return predictor.predict_compatibility(face_data, jewelry_data)