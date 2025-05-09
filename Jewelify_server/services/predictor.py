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
from services.database import get_db_client
from bson import ObjectId
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import urllib.parse  # For URL encoding

# Load environment variables from .env file
load_dotenv()

# Configure logging based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce pymongo logging to INFO to avoid excessive debug logs
logging.getLogger("pymongo").setLevel(logging.INFO)

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

        # Load MobileNetV2 model with a Dense layer to reduce features to 640 per image
        try:
            logger.info("Loading MobileNetV2 model...")
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
                pooling=None,
            )
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(640, activation='relu')(x)  # Reduce to 640 features per image
            self.mobile_net = Model(inputs=base_model.input, outputs=x)
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2 model: {str(e)}")
            raise RuntimeError(f"Failed to load MobileNetV2 model: {str(e)}")

        # Load XGBoost model directly
        try:
            logger.info("Loading XGBoost model...")
            start_time = time.time()
            xgboost_model_path = os.getenv("XGBOOST_MODEL_PATH", "models/xgboost_jewelry_v1.model")
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
            mlp_model_path = os.getenv("MLP_MODEL_PATH", "models/mlp_jewelry_v1.keras")
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
            pairwise_features_path = os.getenv("PAIRWISE_FEATURES_PATH", "models/pairwise_features.npy")
            if not os.path.exists(pairwise_features_path):
                raise FileNotFoundError(f"Pairwise features file not found at {pairwise_features_path}")
            self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
            logger.info(f"Pairwise features loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading pairwise features: {str(e)}")
            raise RuntimeError(f"Failed to load pairwise features: {str(e)}")

        # Updated jewelry categories to match expected values
        self.jewelry_categories = [
            "Necklace with earrings",
            "Earrings only",
            "Necklace only",
            "Bracelet",
            "Ring",
            "Not Assigned"
        ]

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

    async def validate_images(self, face_image: np.ndarray, jewelry_image: np.ndarray, prediction_id: str) -> bool:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            return False
        db = client["jewelify"]
        predictions_collection = db["predictions"]

        logger.info(f"Validating images for prediction {prediction_id}...")
        start_time = time.time()
        try:
            # Validate face image
            is_valid_face, face_message = self.validate_face_image(face_image)
            if not is_valid_face:
                logger.warning(f"Face image validation failed: {face_message}")
                predictions_collection.update_one(
                    {"_id": ObjectId(prediction_id)},
                    {"$set": {"validation_status": "failed"}}
                )
                return False

            # Validate jewelry image
            is_valid_jewelry, jewelry_message = self.validate_image(jewelry_image)
            if not is_valid_jewelry:
                logger.warning(f"Jewelry image validation failed: {jewelry_message}")
                predictions_collection.update_one(
                    {"_id": ObjectId(prediction_id)},
                    {"$set": {"validation_status": "failed"}}
                )
                return False

            # Validation successful
            predictions_collection.update_one(
                {"_id": ObjectId(prediction_id)},
                {"$set": {"validation_status": "completed"}}
            )
            logger.info(f"Validation completed in {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error during validation for prediction {prediction_id}: {str(e)}")
            predictions_collection.update_one(
                {"_id": ObjectId(prediction_id)},
                {"$set": {"validation_status": "failed"}}
            )
            raise

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
            confidence, category = 0.0, "Not Assigned"
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
            confidence, category = 0.0, "Not Assigned"
        logger.info(f"MLP prediction completed in {time.time() - start_time:.2f} seconds")
        return confidence, category

    async def get_recommendations_with_xgboost(self, category: str, top_k: int = 10) -> List[dict]:
        logger.info("Getting XGBoost recommendations...")
        start_time = time.time()
        recommendations = []
        try:
            if category not in self.jewelry_categories:
                logger.warning(f"Category {category} not in jewelry_categories, using fallback")
                category = "Not Assigned"

            # Generate 10 recommendations based on the predicted category
            for i in range(top_k):
                # Encode the URL to handle spaces and special characters
                base_url = "https://jewelify-images.s3.eu-north-1.amazonaws.com/Necklace with earings_sorted_jpg/"
                file_name = f"Necklace with earings_{i}.jpg"
                encoded_file_name = urllib.parse.quote(file_name)
                display_url = f"{base_url}{encoded_file_name}"
                recommendations.append({
                    "name": f"{category}_xgboost_{i}",
                    "category": category,
                    "score": float(100 - i * 5),  # Adjusted scoring for 10 items
                    "display_url": display_url,
                })
        except Exception as e:
            logger.error(f"Error generating XGBoost recommendations: {str(e)}. Returning empty recommendations.")
            recommendations = []
        while len(recommendations) < top_k:
            recommendations.append({
                "name": f"fallback_xgboost_{len(recommendations)}",
                "category": "Not Assigned",
                "score": 0.0,
                "display_url": "",
            })
        logger.info(f"XGBoost recommendations generated in {time.time() - start_time:.2f} seconds")
        return recommendations

    async def get_recommendations_with_mlp(self, category: str, top_k: int = 10) -> List[dict]:
        logger.info("Getting MLP recommendations...")
        start_time = time.time()
        recommendations = []
        try:
            if category not in self.jewelry_categories:
                logger.warning(f"Category {category} not in jewelry_categories, using fallback")
                category = "Not Assigned"

            # Generate 10 recommendations based on the predicted category
            for i in range(top_k):
                # Encode the URL to handle spaces and special characters
                base_url = "https://jewelify-images.s3.eu-north-1.amazonaws.com/Necklace with earings_sorted_jpg/"
                file_name = f"Necklace with earings_{i}.jpg"
                encoded_file_name = urllib.parse.quote(file_name)
                display_url = f"{base_url}{encoded_file_name}"
                recommendations.append({
                    "name": f"{category}_mlp_{i}",
                    "category": category,
                    "score": float(100 - i * 5),  # Adjusted scoring for 10 items
                    "display_url": display_url,
                })
        except Exception as e:
            logger.error(f"Error generating MLP recommendations: {str(e)}. Returning empty recommendations.")
            recommendations = []
        while len(recommendations) < top_k:
            recommendations.append({
                "name": f"fallback_mlp_{len(recommendations)}",
                "category": "Not Assigned",
                "score": 0.0,
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
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            return {}
        db = client["jewelify"]
        predictions_collection = db["predictions"]

        logger.info(f"Starting prediction for both models for prediction {prediction_id}...")
        start_time = time.time()
        try:
            face_features = self.extract_features(face_image) if face_image is not None else None
            jewelry_features = self.extract_features(jewelry_image) if jewelry_image is not None else None
            if face_features is None or jewelry_features is None:
                logger.error("Failed to extract features from images")
                predictions_collection.update_one(
                    {"_id": ObjectId(prediction_id)},
                    {"$set": {"prediction_status": "failed"}}
                )
                return {}

            combined_features = np.concatenate([face_features, jewelry_features])
            logger.info(f"Combined features shape: {combined_features.shape}")  # Should be (1280,)

            # Run prediction tasks
            xgboost_task = self.predict_with_xgboost(combined_features)
            mlp_task = self.predict_with_mlp(combined_features)

            xgboost_result, mlp_result = await asyncio.gather(xgboost_task, mlp_task, return_exceptions=True)

            # Check for exceptions in the prediction tasks
            for i, result in enumerate([xgboost_result, mlp_result]):
                if isinstance(result, Exception):
                    logger.error(f"Prediction task {i} failed with exception: {str(result)}")
                    predictions_collection.update_one(
                        {"_id": ObjectId(prediction_id)},
                        {"$set": {"prediction_status": "failed"}}
                    )
                    raise result

            xgboost_confidence, xgboost_category = xgboost_result
            mlp_confidence, mlp_category = mlp_result

            # Run recommendation tasks based on predicted categories
            xgboost_rec_task = self.get_recommendations_with_xgboost(xgboost_category, top_k=10)
            mlp_rec_task = self.get_recommendations_with_mlp(mlp_category, top_k=10)

            xgboost_recommendations, mlp_recommendations = await asyncio.gather(xgboost_rec_task, mlp_rec_task, return_exceptions=True)

            # Check for exceptions in the recommendation tasks
            for i, result in enumerate([xgboost_recommendations, mlp_recommendations]):
                if isinstance(result, Exception):
                    logger.error(f"Recommendation task {i} failed with exception: {str(result)}")
                    predictions_collection.update_one(
                        {"_id": ObjectId(prediction_id)},
                        {"$set": {"prediction_status": "failed"}}
                    )
                    raise result

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
            predictions_collection.update_one(
                {"_id": ObjectId(prediction_id)},
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
            logger.error(f"Error during prediction for prediction {prediction_id}: {str(e)}")
            predictions_collection.update_one(
                {"_id": ObjectId(prediction_id)},
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