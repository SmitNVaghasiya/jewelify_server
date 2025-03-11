from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from services.predictor import get_predictor, predict_compatibility
from services.database import save_prediction, get_prediction_by_id
from api.dependencies import get_current_user
import os
from fastapi import Form

router = APIRouter(prefix="/predictions", tags=["predictions"])
predictor = get_predictor(
    os.getenv("MODEL_PATH", "rl_jewelry_model.keras"),
    os.getenv("SCALER_PATH", "scaler.pkl"),
    os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")
)

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),  # Receive local path
    jewelry_image_path: str = Form(...),  # Receive local path
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
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded images: {str(e)}")

    try:
        score, category, recommendations = predict_compatibility(predictor, face_data, jewelry_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    if score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Ensure recommendations include score and category
    formatted_recommendations = [
        {
            "name": rec["name"],
            "url": rec.get("url"),
            "score": rec.get("score", score),
            "category": rec.get("category", category)
        }
        for rec in recommendations
    ]

    try:
        prediction_id = save_prediction(
            score,
            category,
            formatted_recommendations,
            str(current_user["_id"]),
            face_image_path,  # Store local path
            jewelry_image_path  # Store local path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prediction: {str(e)}")
    
    return {
        "prediction_id": prediction_id,
        "user_id": str(current_user["_id"]),
        "score": score,
        "category": category,
        "recommendations": formatted_recommendations,
        "face_image_path": face_image_path,  # Return local path
        "jewelry_image_path": jewelry_image_path  # Return local path
    }

@router.get("/get_prediction/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        result = get_prediction_by_id(prediction_id, str(current_user["_id"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if "error" in result:
        status_code = 404 if result["error"] == "Prediction not found" else 500
        raise HTTPException(status_code=status_code, detail=result["error"])
    
    result["user_id"] = str(current_user["_id"])
    return result