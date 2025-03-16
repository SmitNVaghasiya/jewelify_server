import cv2
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle
import logging
from sklearn.neural_network import MLPClassifier
from typing import List, Tuple, Optional
import asyncio
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JewelryPredictor:
    def __init__(self, device: str = "CPU"):
        self.device = device
        # Load Haar Cascade for face detection
        cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade for face detection")
            raise Exception("Failed to load Haar Cascade for face detection")
        self.mobile_net = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        self.xgboost_model = self.load_xgboost_model()
        self.mlp_model = self.load_mlp_model()
        self.jewelry_categories = ["Earring", "Necklace", "Bracelet", "Ring", "Not Assigned"]
        # Cache pairwise features in memory
        self.pairwise_features = self.load_pairwise_features()

    def load_xgboost_model(self) -> xgb.XGBClassifier:
        try:
            logger.info("Loading XGBoost model...")
            start_time = time.time()
            model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            logger.info(f"XGBoost model loaded in {time.time() - start_time:.2f} seconds")
            return model
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            raise

    def load_mlp_model(self) -> MLPClassifier:
        try:
            logger.info("Loading MLP model...")
            start_time = time.time()
            model_path = os.path.join(os.path.dirname(__file__), "mlp_model.pkl")
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            logger.info(f"MLP model loaded in {time.time() - start_time:.2f} seconds")
            return model
        except Exception as e:
            logger.error(f"Error loading MLP model: {str(e)}")
            raise

    def load_pairwise_features(self) -> np.ndarray:
        try:
            logger.info("Loading pairwise features...")
            start_time = time.time()
            features_path = os.path.join(os.path.dirname(__file__), "pairwise_features.npy")
            features = np.load(features_path)
            logger.info(f"Pairwise features loaded in {time.time() - start_time:.2f} seconds")
            return features
        except Exception as e:
            logger.error(f"Error loading pairwise features: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        logger.debug("Preprocessing image...")
        start_time = time.time()
        image = cv2.resize(image, (224, 224))
        image = image.astype("float32")
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        logger.debug(f"Image preprocessed in {time.time() - start_time:.2f} seconds")
        return image

    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        logger.info("Validating image...")
        start_time = time.time()
        if image is None or image.size == 0:
            logger.warning("Validation failed: Image is empty")
            return False, "Image is empty"

        # Check if the image is blank
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 10:
            logger.warning("Validation failed: Image is blank")
            return False, "Image is blank"

        logger.info(f"Image validated in {time.time() - start_time:.2f} seconds")
        return True, "Valid"

    def validate_face_image(self, image: np.ndarray) -> Tuple[bool, str]:
        logger.info("Validating face image...")
        start_time = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            logger.warning("Validation failed: No faces detected")
            return False, "No faces detected in the image"
        logger.info(f"Face image validated in {time.time() - start_time:.2f} seconds")
        return True, "Valid"

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        logger.info("Extracting features...")
        start_time = time.time()
        preprocessed_image = self.preprocess_image(image)
        with tf.device(self.device):
            features = self.mobile_net.predict(preprocessed_image)
        logger.info(f"Features extracted in {time.time() - start_time:.2f} seconds")
        return features.flatten()

    async def predict_with_xgboost(self, features: np.ndarray) -> Tuple[float, str]:
        logger.info("Predicting with XGBoost...")
        start_time = time.time()
        features_2d = features.reshape(1, -1)
        prediction_proba = self.xgboost_model.predict_proba(features_2d)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class] * 100
        category = self.jewelry_categories[predicted_class]
        logger.info(f"XGBoost prediction completed in {time.time() - start_time:.2f} seconds")
        return confidence, category

    async def predict_with_mlp(self, features: np.ndarray) -> Tuple[float, str]:
        logger.info("Predicting with MLP...")
        start_time = time.time()
        features_2d = features.reshape(1, -1)
        prediction_proba = self.mlp_model.predict_proba(features_2d)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class] * 100
        category = self.jewelry_categories[predicted_class]
        logger.info(f"MLP prediction completed in {time.time() - start_time:.2f} seconds")
        return confidence, category

    async def get_recommendations_with_xgboost(self, features: np.ndarray, top_k: int = 3) -> List[dict]:
        logger.info("Getting XGBoost recommendations...")
        start_time = time.time()
        distances = np.linalg.norm(self.pairwise_features - features, axis=1)
        top_k_indices = np.argsort(distances)[:top_k]
        recommendations = []
        for idx in top_k_indices:
            category_idx = int(distances[idx] % len(self.jewelry_categories))
            recommendations.append({
                "name": f"{self.jewelry_categories[category_idx]}_{idx}",
                "category": self.jewelry_categories[category_idx],
                "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{idx}.jpg",
            })
        logger.info(f"XGBoost recommendations generated in {time.time() - start_time:.2f} seconds")
        return recommendations

    async def get_recommendations_with_mlp(self, features: np.ndarray, top_k: int = 3) -> List[dict]:
        logger.info("Getting MLP recommendations...")
        start_time = time.time()
        distances = np.linalg.norm(self.pairwise_features - features, axis=1)
        top_k_indices = np.argsort(distances)[:top_k]
        recommendations = []
        for idx in top_k_indices:
            category_idx = int(distances[idx] % len(self.jewelry_categories))
            recommendations.append({
                "name": f"{self.jewelry_categories[category_idx]}_{idx}",
                "category": self.jewelry_categories[category_idx],
                "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{idx}.jpg",
            })
        logger.info(f"MLP recommendations generated in {time.time() - start_time:.2f} seconds")
        return recommendations

    async def predict_both(
        self,
        face_image: np.ndarray,
        jewelry_image: np.ndarray,
        face_image_path: str,
        jewelry_image_path: str,
    ) -> dict:
        logger.info("Starting prediction for both models...")
        start_time = time.time()
        face_features = self.extract_features(face_image)
        jewelry_features = self.extract_features(jewelry_image)
        combined_features = np.concatenate([face_features, jewelry_features])

        # Run predictions and recommendations in parallel
        tasks = [
            self.predict_with_xgboost(combined_features),
            self.predict_with_mlp(combined_features),
            self.get_recommendations_with_xgboost(combined_features),
            self.get_recommendations_with_mlp(combined_features),
        ]

        results = await asyncio.gather(*tasks)

        xgboost_confidence, xgboost_category = results[0]
        mlp_confidence, mlp_category = results[1]
        xgboost_recommendations = results[2]
        mlp_recommendations = results[3]

        prediction_result = {
            "prediction1": {
                "score": float(xgboost_confidence),
                "category": xgboost_category,
                "recommendations": xgboost_recommendations,
                "feedback_required": True,
            },
            "prediction2": {
                "score": float(mlp_confidence),
                "category": mlp_category,
                "recommendations": mlp_recommendations,
                "feedback_required": True,
            },
            "face_image_path": face_image_path,
            "jewelry_image_path": jewelry_image_path,
        }
        logger.info(f"Prediction for both models completed in {time.time() - start_time:.2f} seconds")
        return prediction_result

if __name__ == "__main__":
    predictor = JewelryPredictor()
    image = cv2.imread("sample_face.jpg")
    jewelry_image = cv2.imread("sample_jewelry.jpg")
    start_time = time.time()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        predictor.predict_both(image, jewelry_image, "sample_face.jpg", "sample_jewelry.jpg")
    )
    print(result)
    print(f"Total time: {time.time() - start_time:.2f} seconds")