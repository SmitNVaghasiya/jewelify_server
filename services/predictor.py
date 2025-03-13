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

class JewelryPredictor:
    def __init__(self, xgboost_model_path, fnn_model_path, scaler_path, pairwise_features_path):
        for path in [xgboost_model_path, fnn_model_path, scaler_path, pairwise_features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        print("Loading XGBoost model...")
        self.xgboost_model = xgb.Booster()
        self.xgboost_model.load_model(xgboost_model_path)

        print("Loading FNN model...")
        self.fnn_model = load_model(fnn_model_path)
        self.img_size = (224, 224)
        self.feature_size = 1280
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

        print("Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("Setting up MobileNetV2 feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        global_avg_layer = GlobalAveragePooling2D()
        reduction_layer = Dense(self.feature_size, activation="relu")
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=reduction_layer(global_avg_layer(base_model.output))
        )

        print("Loading pairwise features...")
        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.pairwise_features = {
            k: self.scaler.transform(np.array(v).reshape(1, -1))
            for k, v in self.pairwise_features.items() if v is not None and v.size == self.feature_size
        }
        self.jewelry_list = list(self.pairwise_features.values())
        self.jewelry_names = list(self.pairwise_features.keys())
        print("Predictor initialized successfully!")

    def extract_features(self, img_data):
        try:
            img = image.load_img(BytesIO(img_data), target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def combine_features(self, face_features, jewelry_features):
        return np.concatenate((face_features, jewelry_features), axis=1)

    def predict_with_xgboost(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        dmatrix = xgb.DMatrix(combined_features)
        score = self.xgboost_model.predict(dmatrix)[0] * 100.0  # Scale to 0-100
        category = self.compute_category(score)
        return score, category

    def predict_with_fnn(self, face_features, jewelry_features):
        combined_features = self.combine_features(face_features, jewelry_features)
        with tf.device(self.device):
            score = self.fnn_model.predict(combined_features, verbose=0)[0][0] * 100.0  # Scale to 0-100
        category = self.compute_category(score)
        return score, category

    def get_recommendations_with_xgboost(self, face_features):
        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        combined_features = np.vstack(combined_features)
        dmatrix = xgb.DMatrix(combined_features)
        scores = self.xgboost_model.predict(dmatrix) * 100.0
        indices = np.argsort(scores)[::-1][:20]  # Top 20 recommendations
        recommendations = [self.jewelry_names[i] for i in indices]
        rec_scores = scores[indices]
        rec_categories = [self.compute_category(score) for score in rec_scores]
        return [{"name": name, "score": round(score, 2), "category": category} for name, score, category in zip(recommendations, rec_scores, rec_categories)]

    def get_recommendations_with_fnn(self, face_features):
        combined_features = [self.combine_features(face_features, jewel_feature) for jewel_feature in self.jewelry_list]
        combined_features = np.vstack(combined_features)
        with tf.device(self.device):
            scores = self.fnn_model.predict(combined_features, verbose=0).flatten() * 100.0
        indices = np.argsort(scores)[::-1][:20]  # Top 20 recommendations
        recommendations = [self.jewelry_names[i] for i in indices]
        rec_scores = scores[indices]
        rec_categories = [self.compute_category(score) for score in rec_scores]
        return [{"name": name, "score": round(score, 2), "category": category} for name, score, category in zip(recommendations, rec_scores, rec_categories)]

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

def get_predictor(xgboost_model_path, fnn_model_path, scaler_path, pairwise_features_path):
    try:
        predictor = JewelryPredictor(xgboost_model_path, fnn_model_path, scaler_path, pairwise_features_path)
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
    xgboost_score, xgboost_category = predictor.predict_with_xgboost(face_features, jewelry_features)
    fnn_score, fnn_category = predictor.predict_with_fnn(face_features, jewelry_features)
    xgboost_recommendations = predictor.get_recommendations_with_xgboost(face_features)
    fnn_recommendations = predictor.get_recommendations_with_fnn(face_features)
    return xgboost_score, xgboost_category, xgboost_recommendations, fnn_score, fnn_category, fnn_recommendations