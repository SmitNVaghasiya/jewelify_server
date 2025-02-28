import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from predictor import get_predictor, predict_compatibility
from pymongo import MongoClient
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global MongoDB client
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("🚨 MONGO_URI not found in environment variables")
    client = None
else:
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')  # Test connection
        logger.info("✅ Successfully connected to MongoDB Atlas!")
    except Exception as e:
        logger.error(f"🚨 Failed to connect to MongoDB Atlas: {e}")
        client = None

def rebuild_client():
    """Attempt to rebuild the MongoDB client if it is None or fails."""
    global client, MONGO_URI
    if not MONGO_URI:
        logger.error("🚨 Cannot rebuild client: MONGO_URI not found")
        return False
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')  # Test connection
        logger.info("✅ Successfully rebuilt MongoDB client")
        return True
    except Exception as e:
        logger.error(f"🚨 Failed to rebuild MongoDB client: {e}")
        return False

def save_prediction(score, category, recommendations):
    """Save prediction to MongoDB and return the inserted _id as a string"""
    global client
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot save prediction")
            return None

    try:
        db = client["jewelify"]
        collection = db["recommendations"]
        prediction = {
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(prediction)
        logger.info(f"✅ Saved prediction with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_all_predictions():
    """Retrieve all predictions with image URLs"""
    global client
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot retrieve predictions")
            return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        # Get all predictions
        predictions = list(predictions_collection.find().sort("timestamp", -1))
        if not predictions:
            logger.warning("⚠️ No predictions found")
            return {"error": "No predictions found"}

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for name in recommendations:
                image_doc = images_collection.find_one({"name": name})
                if image_doc:
                    image_data.append({"name": name, "url": image_doc["url"]})
                else:
                    image_data.append({"name": name, "url": None})

            results.append({
                "id": str(prediction["_id"]),
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "timestamp": prediction["timestamp"]
            })

        logger.info(f"✅ Retrieved {len(results)} predictions")
        return results
    except Exception as e:
        logger.error(f"❌ Error retrieving predictions from MongoDB: {e}")
        return {"error": str(e)}

# Define paths using environment variables with defaults
MODEL_PATH = os.getenv("MODEL_PATH", "rl_jewelry_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
PAIRWISE_FEATURES_PATH = os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")

# Initialize predictor globally
predictor = get_predictor(MODEL_PATH, SCALER_PATH, PAIRWISE_FEATURES_PATH)

# FastAPI app
app = FastAPI()

@app.get('/')
async def home():
    return {"Message": "Welcome to Jewelify home page"}

@app.post("/predict")
async def predict(
    face: UploadFile = File(...),  # Required file upload
    jewelry: UploadFile = File(...)  # Required file upload
):
    global predictor

    # Reinitialize predictor if None
    if predictor is None:
        print("⚠️ Predictor is None, attempting to reinitialize...")
        try:
            predictor = get_predictor(MODEL_PATH, SCALER_PATH, PAIRWISE_FEATURES_PATH)
            if predictor is None:
                raise Exception("Predictor reinitialization failed")
        except Exception as e:
            print(f"🚨 Failed to reinitialize JewelryRLPredictor: {e}")
            raise HTTPException(status_code=500, detail="Model is not loaded properly")

    # Validate file types
    if not face.content_type.startswith('image/') or not jewelry.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")

    # Read uploaded image files
    try:
        face_data = await face.read()
        jewelry_data = await jewelry.read()
    except Exception as e:
        print(f"❌ Error reading uploaded files: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded images")

    # Perform prediction
    score, category, recommendations = predict_compatibility(predictor, face_data, jewelry_data)
    if score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Save to MongoDB and get prediction_id
    prediction_id = save_prediction(score, category, recommendations)
    if prediction_id is None:
        print("⚠️ Failed to save prediction to MongoDB, but returning response anyway")

    return {
        "score": score,
        "category": category,
        "recommendations": recommendations,
        "prediction_id": prediction_id
    }

@app.get("/get_predictions")
async def get_predictions():
    """Retrieve all predictions with image URLs"""
    result = get_all_predictions()
    if "error" in result:
        status_code = 500 if result["error"] != "No predictions found" else 404
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT; default to 5000
    uvicorn.run(app, host="0.0.0.0", port=port)