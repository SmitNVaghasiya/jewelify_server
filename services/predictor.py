# services/predictor.py
import cv2
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle
import logging
import os
from typing import List, Tuple, Optional
import asyncio
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JewelryPredictor:
    def __init__(self, device: str = "CPU"):
        self.device = device
        
        # Load Haar Cascade file
        root_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from services/ to root
        cascade_path = os.getenv("HAAR_CASCADE_PATH", os.path.join(root_dir, "haarcascade_frontalface_default.xml"))
        if not os.path.exists(cascade_path):
            logger.warning(f"Haar Cascade file not found at {cascade_path}. Face validation will be skipped.")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning(f"Failed to load Haar Cascade file at {cascade_path}. Face validation will be skipped.")
                self.face_cascade = None
            else:
                logger.info("Haar Cascade loaded successfully")

        # Load MobileNetV2 model
        try:
            self.mobile_net = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2 model: {str(e)}")
            raise RuntimeError(f"Failed to load MobileNetV2 model: {str(e)}")

        # Load XGBoost and MLP models
        self.xgboost_model = self.load_xgboost_model()
        self.mlp_model = self.load_mlp_model()
        self.jewelry_categories = ["Earring", "Necklace", "Bracelet", "Ring", "Not Assigned"]
        self.pairwise_features = self.load_pairwise_features()

    def load_xgboost_model(self) -> xgb.Booster:
        """Load the XGBoost model using the native XGBoost method."""
        try:
            logger.info("Loading XGBoost model...")
            start_time = time.time()
            root_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.getenv("XGBOOST_MODEL_PATH", os.path.join(root_dir, "xgboost_jewelry_v1.model"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"XGBoost model file not found at {model_path}")
            
            # Use XGBoost's native load_model method
            model = xgb.Booster()
            model.load_model(model_path)
            logger.info(f"XGBoost model loaded in {time.time() - start_time:.2f} seconds")
            return model
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}. Ensure the model file is valid and saved with xgb.Booster.save_model().")
            raise RuntimeError(f"Failed to load XGBoost model: {str(e)}")

    def load_mlp_model(self) -> Optional[tf.keras.Model]:
        try:
            logger.info("Loading MLP model...")
            start_time = time.time()
            root_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.getenv("MLP_MODEL_PATH", os.path.join(root_dir, "mlp_jewelry_v1.keras"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MLP model file not found at {model_path}")
            model = tf.keras.models.load_model(model_path)
            logger.info(f"MLP model loaded in {time.time() - start_time:.2f} seconds")
            return model
        except Exception as e:
            logger.error(f"Error loading MLP model: {str(e)}")
            raise RuntimeError(f"Failed to load MLP model: {str(e)}")

    def load_pairwise_features(self) -> np.ndarray:
        try:
            logger.info("Loading pairwise features...")
            start_time = time.time()
            root_dir = os.path.dirname(os.path.dirname(__file__))
            features_path = os.getenv("PAIRWISE_FEATURES_PATH", os.path.join(root_dir, "pairwise_features.npy"))
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Pairwise features file not found at {features_path}")
            features = np.load(features_path)
            logger.info(f"Pairwise features loaded in {time.time() - start_time:.2f} seconds")
            return features
        except Exception as e:
            logger.error(f"Error loading pairwise features: {str(e)}")
            raise RuntimeError(f"Failed to load pairwise features: {str(e)}")

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
        if self.face_cascade is None:
            logger.warning("Haar Cascade not loaded, skipping face validation")
            return True, "Validation skipped (Haar Cascade not loaded)"
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
        try:
            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            prediction = self.xgboost_model.predict(dmatrix)[0]
            # Assuming a classification model with class indices; adjust if it's regression
            predicted_class = int(round(prediction)) if 0 <= prediction < len(self.jewelry_categories) else 0
            confidence = 100.0  # Placeholder; adjust based on your model's output (e.g., probability if available)
            category = self.jewelry_categories[predicted_class]
        except Exception as e:
            logger.error(f"Error during XGBoost prediction: {str(e)}. Returning fallback values.")
            confidence, category = 50.0, "Neutral"
        logger.info(f"XGBoost prediction completed in {time.time() - start_time:.2f} seconds")
        return confidence, category

    async def predict_with_mlp(self, features: np.ndarray) -> Tuple[float, str]:
        logger.info("Predicting with MLP...")
        start_time = time.time()
        features_2d = features.reshape(1, -1)
        try:
            prediction_proba = self.mlp_model.predict(features_2d, verbose=0)[0]
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class] * 100
            category = self.jewelry_categories[predicted_class]
        except Exception as e:
            logger.error(f"Error during MLP prediction: {str(e)}. Returning fallback values.")
            confidence, category = 50.0, "Neutral"
        logger.info(f"MLP prediction completed in {time.time() - start_time:.2f} seconds")
        return confidence, category

    async def get_recommendations_with_xgboost(self, features: np.ndarray, top_k: int = 3) -> List[dict]:
        logger.info("Getting XGBoost recommendations...")
        start_time = time.time()
        recommendations = []
        try:
            if self.pairwise_features is not None and len(self.pairwise_features) > 0:
                # Use distance-based recommendation (consistent with your previous logic)
                distances = np.linalg.norm(self.pairwise_features - features, axis=1)
                top_k_indices = np.argsort(distances)[:top_k]
                for idx in top_k_indices:
                    category_idx = int(distances[idx] % len(self.jewelry_categories))
                    recommendations.append({
                        "name": f"{self.jewelry_categories[category_idx]}_{idx}",
                        "category": self.jewelry_categories[category_idx],
                        "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{idx}.jpg",
                    })
            else:
                logger.warning("No pairwise features available, returning empty recommendations.")
        except Exception as e:
            logger.error(f"Error generating XGBoost recommendations: {str(e)}. Returning empty recommendations.")
        # Ensure top_k recommendations (fill with fallback if needed)
        while len(recommendations) < top_k:
            recommendations.append({
                "name": f"fallback_xgboost_{len(recommendations)}",
                "category": "Neutral",
                "display_url": "",
            })
        logger.info(f"XGBoost recommendations generated in {time.time() - start_time:.2f} seconds")
        return recommendations

    async def get_recommendations_with_mlp(self, features: np.ndarray, top_k: int = 3) -> List[dict]:
        logger.info("Getting MLP recommendations...")
        start_time = time.time()
        recommendations = []
        try:
            if self.pairwise_features is not None and len(self.pairwise_features) > 0:
                distances = np.linalg.norm(self.pairwise_features - features, axis=1)
                top_k_indices = np.argsort(distances)[:top_k]
                for idx in top_k_indices:
                    category_idx = int(distances[idx] % len(self.jewelry_categories))
                    recommendations.append({
                        "name": f"{self.jewelry_categories[category_idx]}_{idx}",
                        "category": self.jewelry_categories[category_idx],
                        "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{idx}.jpg",
                    })
            else:
                logger.warning("No pairwise features available, returning empty recommendations.")
        except Exception as e:
            logger.error(f"Error generating MLP recommendations: {str(e)}. Returning empty recommendations.")
        # Ensure top_k recommendations (fill with fallback if needed)
        while len(recommendations) < top_k:
            recommendations.append({
                "name": f"fallback_mlp_{len(recommendations)}",
                "category": "Neutral",
                "display_url": "",
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
        face_features = self.extract_features(face_image) if face_image is not None else None
        jewelry_features = self.extract_features(jewelry_image) if jewelry_image is not None else None
        combined_features = np.concatenate([face_features, jewelry_features]) if face_features is not None and jewelry_features is not None else None

        # Run predictions and recommendations in parallel with fallback
        tasks = [
            self.predict_with_xgboost(combined_features) if combined_features is not None else asyncio.Future(),
            self.predict_with_mlp(combined_features) if combined_features is not None else asyncio.Future(),
            self.get_recommendations_with_xgboost(face_features) if face_features is not None else asyncio.Future(),
            self.get_recommendations_with_mlp(face_features) if face_features is not None else asyncio.Future(),
        ]

        # Replace Future objects with fallback values
        for i, task in enumerate(tasks):
            if isinstance(task, asyncio.Future):
                tasks[i] = asyncio.Future()
                tasks[i].set_result((50.0, "Neutral") if i < 2 else [{"name": f"fallback_{i}", "category": "Neutral", "display_url": ""} for _ in range(3)])

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