import cv2
import numpy as np
import tensorflow as tf
import xgboost as xgb
import logging
import os
from typing import List, Tuple, Optional
import asyncio
from dotenv import load_dotenv
import time
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB client setup for updating prediction status
mongo_uri = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(mongo_uri)
db = client["jewelry_db"]
predictions_collection = db["predictions"]

class JewelryPredictor:
    def __init__(self, device: str = "CPU"):
        self.device = device
        
        # Load Haar Cascade file
        cascade_path = os.getenv("HAAR_CASCADE_PATH", "haarcascade_frontalface_default.xml")
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

        # Load XGBoost model directly
        try:
            logger.info("Loading XGBoost model...")
            start_time = time.time()
            xgboost_model_path = os.getenv("XGBOOST_MODEL_PATH", "xgboost_jewelry_v1.model")
            if not os.path.exists(xgboost_model_path):
                raise FileNotFoundError(f"XGBoost model file not found at {xgboost_model_path}")
            self.xgboost_model = xgb.Booster()
            self.xgboost_model.load_model(xgboost_model_path)
            logger.info(f"XGBoost model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            raise RuntimeError(f"Failed to load XGBoost model: {str(e)}")

        # Load MLP model directly
        try:
            logger.info("Loading MLP model...")
            start_time = time.time()
            mlp_model_path = os.getenv("MLP_MODEL_PATH", "mlp_jewelry_v1.keras")
            if not os.path.exists(mlp_model_path):
                raise FileNotFoundError(f"MLP model file not found at {mlp_model_path}")
            self.mlp_model = tf.keras.models.load_model(mlp_model_path)
            logger.info(f"MLP model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading MLP model: {str(e)}")
            raise RuntimeError(f"Failed to load MLP model: {str(e)}")

        # Load pairwise features directly
        try:
            logger.info("Loading pairwise features...")
            start_time = time.time()
            pairwise_features_path = os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")
            if not os.path.exists(pairwise_features_path):
                raise FileNotFoundError(f"Pairwise features file not found at {pairwise_features_path}")
            self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
            logger.info(f"Pairwise features loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading pairwise features: {str(e)}")
            raise RuntimeError(f"Failed to load pairwise features: {str(e)}")

        self.jewelry_categories = ["Earring", "Necklace", "Bracelet", "Ring", "Not Assigned"]

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

    async def validate_images(self, face_image: np.ndarray, jewelry_image: np.ndarray, prediction_id: str):
        logger.info(f"Validating images for prediction {prediction_id}...")
        start_time = time.time()
        try:
            # Validate face image
            is_valid_face, face_message = self.validate_face_image(face_image)
            if not is_valid_face:
                logger.warning(f"Face image validation failed: {face_message}")
                await predictions_collection.update_one(
                    {"_id": prediction_id},
                    {"$set": {"validation_status": "failed"}}
                )
                return

            # Validate jewelry image
            is_valid_jewelry, jewelry_message = self.validate_image(jewelry_image)
            if not is_valid_jewelry:
                logger.warning(f"Jewelry image validation failed: {jewelry_message}")
                await predictions_collection.update_one(
                    {"_id": prediction_id},
                    {"$set": {"validation_status": "failed"}}
                )
                return

            # Validation successful
            await predictions_collection.update_one(
                {"_id": prediction_id},
                {"$set": {"validation_status": "completed"}}
            )
            logger.info(f"Validation completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            await predictions_collection.update_one(
                {"_id": prediction_id},
                {"$set": {"validation_status": "failed"}}
            )

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
            predicted_class = int(round(prediction)) if 0 <= prediction < len(self.jewelry_categories) else 0
            confidence = 100.0  # Placeholder; adjust based on your model's output
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
                if isinstance(self.pairwise_features, dict):
                    distances = {key: np.linalg.norm(val - features) for key, val in self.pairwise_features.items()}
                    sorted_items = sorted(distances.items(), key=lambda x: x[1])[:top_k]
                    for name, _ in sorted_items:
                        category_idx = self.jewelry_categories.index(name.split('_')[0]) if '_' in name else 0
                        recommendations.append({
                            "name": name,
                            "category": self.jewelry_categories[category_idx],
                            "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{name.split('_')[-1]}.jpg",
                        })
                else:
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
                if isinstance(self.pairwise_features, dict):
                    distances = {key: np.linalg.norm(val - features) for key, val in self.pairwise_features.items()}
                    sorted_items = sorted(distances.items(), key=lambda x: x[1])[:top_k]
                    for name, _ in sorted_items:
                        category_idx = self.jewelry_categories.index(name.split('_')[0]) if '_' in name else 0
                        recommendations.append({
                            "name": name,
                            "category": self.jewelry_categories[category_idx],
                            "display_url": f"https://jewelify-server.onrender.com/static/images/{self.jewelry_categories[category_idx].lower()}_{name.split('_')[-1]}.jpg",
                        })
                else:
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
        prediction_id: str,
    ) -> dict:
        logger.info(f"Starting prediction for both models for prediction {prediction_id}...")
        start_time = time.time()
        try:
            face_features = self.extract_features(face_image) if face_image is not None else None
            jewelry_features = self.extract_features(jewelry_image) if jewelry_image is not None else None
            combined_features = np.concatenate([face_features, jewelry_features]) if face_features is not None and jewelry_features is not None else None

            tasks = [
                self.predict_with_xgboost(combined_features) if combined_features is not None else asyncio.Future(),
                self.predict_with_mlp(combined_features) if combined_features is not None else asyncio.Future(),
                self.get_recommendations_with_xgboost(face_features) if face_features is not None else asyncio.Future(),
                self.get_recommendations_with_mlp(face_features) if face_features is not None else asyncio.Future(),
            ]

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
                    "overall_feedback": 0.5
                },
                "prediction2": {
                    "score": float(mlp_confidence),
                    "category": mlp_category,
                    "recommendations": mlp_recommendations,
                    "feedback_required": True,
                    "overall_feedback": 0.5
                },
                "face_image_path": face_image_path,
                "jewelry_image_path": jewelry_image_path,
            }

            # Update the database with the prediction result and status
            await predictions_collection.update_one(
                {"_id": prediction_id},
                {
                    "$set": {
                        "prediction1": prediction_result["prediction1"],
                        "prediction2": prediction_result["prediction2"],
                        "face_image_path": prediction_result["face_image_path"],
                        "jewelry_image_path": prediction_result["jewelry_image_path"],
                        "prediction_status": "completed"
                    }
                }
            )
            logger.info(f"Prediction for both models completed in {time.time() - start_time:.2f} seconds")
            return prediction_result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            await predictions_collection.update_one(
                {"_id": prediction_id},
                {"$set": {"prediction_status": "failed"}}
            )
            raise

if __name__ == "__main__":
    predictor = JewelryPredictor()
    image = cv2.imread("sample_face.jpg")
    jewelry_image = cv2.imread("sample_jewelry.jpg")
    start_time = time.time()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        predictor.predict_both(image, jewelry_image, "sample_face.jpg", "sample_jewelry.jpg", "test_id")
    )
    print(result)
    print(f"Total time: {time.time() - start_time:.2f} seconds")