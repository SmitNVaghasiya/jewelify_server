import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from predictor import get_predictor, predict_compatibility
from db import save_prediction, get_latest_prediction
import ssl

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
            return JSONResponse(content={"error": "Model is not loaded properly"}, status_code=500)

    # Read uploaded image files
    try:
        face_data = await face.read()
        jewelry_data = await jewelry.read()
    except Exception as e:
        print(f"❌ Error reading uploaded files: {e}")
        return JSONResponse(content={"error": "Failed to read uploaded images"}, status_code=400)

    # Perform prediction
    score, category, recommendations = predict_compatibility(predictor, face_data, jewelry_data)
    if score is None:
        return JSONResponse(content={"error": "Prediction failed"}, status_code=500)

    # Save to MongoDB and get prediction_id
    prediction_id = save_prediction(float(score), category, recommendations)
    if prediction_id is None:
        print("⚠️ Failed to save prediction to MongoDB, but returning response anyway")

    return {
        "score": float(score),
        "category": category,
        "recommendations": recommendations,
        "prediction_id": prediction_id  # Now includes the MongoDB _id as a string or None
    }

@app.get("/get_predictions")
async def get_predictions():
    """Retrieve the latest prediction with image URLs"""
    result = get_latest_prediction()
    if result is None:
        return JSONResponse(content={"error": "No predictions found or database error"}, status_code=500)
    return result

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT; default to 5000
    uvicorn.run(app, host="0.0.0.0", port=port)