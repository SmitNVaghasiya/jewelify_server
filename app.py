import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from predictor import get_predictor, predict_compatibility
from db import save_prediction, get_prediction_by_id, get_all_predictions
from bson import ObjectId

# Load environment variables
load_dotenv()

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
        "prediction_id": prediction_id,
        "score": score,
        "category": category,
        "recommendations": recommendations,
    }

@app.get("/get_prediction/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Retrieve a single prediction by ID"""
    try:
        result = get_prediction_by_id(prediction_id)
        if "error" in result:
            status_code = 404 if result["error"] == "Prediction not found" else 500
            raise HTTPException(status_code=status_code, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid prediction ID: {str(e)}")

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