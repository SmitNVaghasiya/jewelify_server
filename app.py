import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
import pickle
import logging
import base64
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file (for local use)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# File paths from environment variables with defaults
MODEL_PATH = os.getenv("MODEL_PATH", "rl_jewelry_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
PAIRWISE_FEATURES_PATH = os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")

# FastAPI app setup
app = FastAPI(
    title="Jewelry Compatibility Predictor",
    description="Predict compatibility between base64-encoded face and jewelry images and get top 10 recommendations.",
    version="1.0.0"
)

# Input model for API
class PredictInput(BaseModel):
    face_base64: str
    jewelry_base64: str

# Predictor class
class JewelryRLPredictor:
    def __init__(self, model_path, scaler_path, pairwise_features_path):
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Checking for model at: {model_path}")
        logger.info(f"Checking for scaler at: {scaler_path}")
        logger.info(f"Checking for pairwise features at: {pairwise_features_path}")
        
        missing_files = [p for p in [model_path, scaler_path, pairwise_features_path] if not os.path.exists(p)]
        if missing_files:
            error_msg = f"Missing files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Loading model...")
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        
        logger.info("Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Initializing feature extractor...")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
        
        logger.info("Loading pairwise features...")
        self.pairwise_features = np.load(pairwise_features_path, allow_pickle=True).item()
        self.jewelry_names = list(self.pairwise_features.keys())
        self.jewelry_features = np.array(list(self.pairwise_features.values()))

    def extract_features(self, img_data):
        try:
            img = Image.open(BytesIO(img_data)).convert("RGB").resize(self.img_size, Image.Resampling.LANCZOS)
            img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
            features = self.feature_extractor.predict(img_array, verbose=0)
            return self.scaler.transform(features)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    def predict_compatibility(self, face_data, jewel_data):
        try:
            face_features = self.extract_features(face_data)
            jewel_features = self.extract_features(jewel_data)
            
            similarity = np.dot(face_features, jewel_features.T).flatten()[0]
            score = (similarity + 1) / 2.0
            score = max(0, min(score, 1))
            category = (
                "Very Good" if score >= 0.8 else
                "Good" if score >= 0.6 else
                "Neutral" if score >= 0.4 else
                "Bad" if score >= 0.2 else
                "Very Bad"
            )
            
            recommendation_scores = (np.dot(face_features, self.jewelry_features.T).flatten() + 1) / 2.0
            recommendation_scores = np.clip(recommendation_scores, 0, 1)
            top_10_indices = np.argsort(recommendation_scores)[::-1][:10]
            recommendations = [
                {"name": self.jewelry_names[i], "score": float(recommendation_scores[i])}
                for i in top_10_indices
            ]
            
            return score, category, recommendations
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")

# Initialize predictor
try:
    predictor = JewelryRLPredictor(MODEL_PATH, SCALER_PATH, PAIRWISE_FEATURES_PATH)
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.post("/predict", summary="Predict jewelry compatibility with base64 images")
async def predict(request: Request, input_data: PredictInput):
    """Predict compatibility and return top 10 jewelry recommendations using base64-encoded images."""
    try:
        face_base64 = input_data.face_base64
        jewelry_base64 = input_data.jewelry_base64
        
        if not face_base64 or not jewelry_base64:
            raise HTTPException(status_code=400, detail="Both face and jewelry base64 strings are required")
        
        logger.info("Decoding base64 images...")
        try:
            face_data = base64.b64decode(face_base64)
            jewelry_data = base64.b64decode(jewelry_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 string provided")
        
        logger.info(f"Face data size: {len(face_data)} bytes, Jewelry data size: {len(jewelry_data)} bytes")
        
        if not face_data or not jewelry_data:
            raise HTTPException(status_code=400, detail="Decoded images are empty or invalid")

        score, category, recommendations = predictor.predict_compatibility(face_data, jewelry_data)
        
        logger.info("Prediction successful")
        return {"score": score, "category": category, "recommendations": recommendations}
    
    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)