from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from services.predictor import get_predictor, predict_both
from services.database import save_prediction, get_prediction_by_id, save_review
from api.dependencies import get_current_user
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/predictions", tags=["predictions"])
predictor = get_predictor(
    os.getenv("XGBOOST_MODEL_PATH", "xgboost_jewelry_v1.model"),
    os.getenv("FNN_MODEL_PATH", "FNN_improved_v5.keras"),
    os.getenv("XGBOOST_SCALER_PATH", "scaler_xgboost_v1.pkl"),
    os.getenv("FNN_SCALER_PATH", "FNN_improved_v5_scaler.pkl"),
    os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy"),
    os.getenv("FACE_FEATURES_PATH", "face_features.npy"),
    os.getenv("EARRING_FEATURES_PATH", "earrings_features.npy"),
    os.getenv("NECKLACE_FEATURES_PATH", "necklace_features.npy")
)

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),
    jewelry_image_path: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    global predictor
    if predictor is None:
        predictor = get_predictor(
            os.getenv("XGBOOST_MODEL_PATH", "xgboost_jewelry_v1.model"),
            os.getenv("FNN_MODEL_PATH", "FNN_improved_v5.keras"),
            os.getenv("XGBOOST_SCALER_PATH", "scaler_xgboost_v1.pkl"),
            os.getenv("FNN_SCALER_PATH", "FNN_improved_v5_scaler.pkl"),
            os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy"),
            os.getenv("FACE_FEATURES_PATH", "face_features.npy"),
            os.getenv("EARRING_FEATURES_PATH", "earrings_features.npy"),
            os.getenv("NECKLACE_FEATURES_PATH", "necklace_features.npy")
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
        xgboost_score, xgboost_category, xgboost_recommendations, fnn_score, fnn_category, fnn_recommendations = predict_both(predictor, face_data, jewelry_data)
        if xgboost_category in ["Invalid face image", "Invalid jewelry image"] or fnn_category in ["Invalid face image", "Invalid jewelry image"]:
            if "face" in xgboost_category or "face" in fnn_category:
                raise HTTPException(status_code=400, detail="Uploaded face image is invalid")
            if "jewelry" in xgboost_category or "jewelry" in fnn_category:
                raise HTTPException(status_code=400, detail="Uploaded jewelry image is invalid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    if xgboost_score is None or fnn_score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    try:
        prediction_id = save_prediction(
            xgboost_score, xgboost_category, xgboost_recommendations,
            fnn_score, fnn_category, fnn_recommendations,
            str(current_user["_id"]),
            face_image_path,
            jewelry_image_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prediction: {str(e)}")

    return {
        "prediction_id": prediction_id,
        "user_id": str(current_user["_id"]),
        "prediction1": {
            "score": round(xgboost_score, 2),
            "category": xgboost_category,
            "recommendations": xgboost_recommendations
        },
        "prediction2": {
            "score": round(fnn_score, 2),
            "category": fnn_category,
            "recommendations": fnn_recommendations
        },
        "face_image_path": face_image_path,
        "jewelry_image_path": jewelry_image_path
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

@router.post("/feedback/recommendation")
async def submit_recommendation_feedback(
    prediction_id: str = Form(...),
    model_type: str = Form(...),  # "prediction1" or "prediction2"
    recommendation_name: str = Form(...),
    review: str = Form(...),  # "yes", "no", "neutral"
    current_user: dict = Depends(get_current_user)
):
    valid_reviews = {"yes", "no", "neutral"}
    if review not in valid_reviews:
        raise HTTPException(status_code=400, detail="Review must be 'yes', 'no', or 'neutral'")

    if model_type not in ["prediction1", "prediction2"]:
        raise HTTPException(status_code=400, detail="Model type must be 'prediction1' or 'prediction2'")

    success = save_review(
        str(current_user["_id"]),
        prediction_id,
        model_type,
        recommendation_name,
        review,
        feedback_type="recommendation"
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save review")

    return {"message": "Recommendation review saved successfully"}

@router.post("/feedback/prediction")
async def submit_prediction_feedback(
    prediction_id: str = Form(...),
    model_type: str = Form(...),  # "prediction1" or "prediction2"
    review: str = Form(...),  # "yes", "no", "neutral"
    current_user: dict = Depends(get_current_user)
):
    valid_reviews = {"yes", "no", "neutral"}
    if review not in valid_reviews:
        raise HTTPException(status_code=400, detail="Review must be 'yes', 'no', or 'neutral'")

    if model_type not in ["prediction1", "prediction2"]:
        raise HTTPException(status_code=400, detail="Model type must be 'prediction1' or 'prediction2'")

    success = save_review(
        str(current_user["_id"]),
        prediction_id,
        model_type,
        recommendation_name=None,
        review=review,
        feedback_type="prediction"
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save review")

    return {"message": "Prediction review saved successfully"}