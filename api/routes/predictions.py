from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from services.predictor import get_predictor, predict_compatibility
from services.database import save_prediction
from api.dependencies import get_current_user
import os
import logging

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)

predictor = get_predictor(
    os.getenv("MODEL_PATH", "rl_jewelry_model.keras"),
    os.getenv("SCALER_PATH", "scaler.pkl"),
    os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")
)

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    global predictor
    if predictor is None:
        predictor = get_predictor(
            os.getenv("MODEL_PATH", "rl_jewelry_model.keras"),
            os.getenv("SCALER_PATH", "scaler.pkl"),
            os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")
        )
        if predictor is None:
            raise HTTPException(status_code=500, detail="Model is not loaded properly")

    if not face.content_type.startswith('image/') or not jewelry.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")

    try:
        face_data = await face.read()
        jewelry_data = await jewelry.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded images: " + str(e))

    try:
        score, category, recommendations = predict_compatibility(predictor, face_data, jewelry_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))
    
    if score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    try:
        prediction_id = save_prediction(score, category, recommendations, str(current_user["_id"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save prediction: " + str(e))
    
    return {
        "prediction_id": prediction_id,
        "score": score,
        "category": category,  # This will now contain Unicode emoji codes
        "recommendations": recommendations,
    }

@router.get("/get_predictions")
async def get_predictions(current_user: dict = Depends(get_current_user)):
    try:
        result = get_all_predictions(current_user["_id"])
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error: " + str(e))
    
    if "error" in result:
        status_code = 500 if result["error"] != "No predictions found" else 404
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result

@router.get("/get_predictions")
async def get_predictions():
    from services.database import get_all_predictions
    try:
        result = get_all_predictions()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error: " + str(e))
    
    if "error" in result:
        status_code = 500 if result["error"] != "No predictions found" else 404
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result