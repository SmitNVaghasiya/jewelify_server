from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from services.predictor import get_predictor, predict_both
from services.database import save_prediction, get_prediction_by_id, save_review
from api.dependencies import get_current_user
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),
    jewelry_image_path: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            predictor = get_predictor(
                os.getenv("XGBOOST_MODEL_PATH", "xgboost_jewelry_v1.model"),
                os.getenv("MLP_MODEL_PATH", "mlp_jewelry_v1.keras"),
                os.getenv("XGBOOST_SCALER_PATH", "scaler_xgboost_v1.pkl"),
                os.getenv("MLP_SCALER_PATH", "scaler_mlp_v1.pkl"),
                os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy"),
            )
            if predictor is None:
                raise ValueError("Model is not loaded properly")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to load model after {max_retries} attempts: {str(e)}")
            time.sleep(retry_delay)

    if not face.content_type.startswith('image/') or not jewelry.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")

    try:
        face_data = await face.read()
        jewelry_data = await jewelry.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded images: {str(e)}")

    try:
        xgboost_score, xgboost_category, xgboost_recommendations, mlp_score, mlp_category, mlp_recommendations = predict_both(predictor, face_data, jewelry_data)
        if xgboost_category in ["Invalid face image", "Invalid jewelry image"] or mlp_category in ["Invalid face image", "Invalid jewelry image"]:
            if "face" in xgboost_category.lower() or "face" in mlp_category.lower():
                raise HTTPException(status_code=400, detail="Uploaded face image is invalid")
            if "jewelry" in xgboost_category.lower() or "jewelry" in mlp_category.lower():
                raise HTTPException(status_code=400, detail="Uploaded jewelry image is invalid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    if xgboost_score is None or mlp_score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Ensure exactly 10 recommendations per model (already handled by predictor, but as a safeguard)
    if len(xgboost_recommendations) < 10:
        for i in range(len(xgboost_recommendations), 10):
            xgboost_recommendations.append({"name": f"fallback_xgboost_{i}", "score": 50.0, "category": "Neutral"})
    else:
        xgboost_recommendations = xgboost_recommendations[:10]

    if len(mlp_recommendations) < 10:
        for i in range(len(mlp_recommendations), 10):
            mlp_recommendations.append({"name": f"fallback_mlp_{i}", "score": 50.0, "category": "Neutral"})
    else:
        mlp_recommendations = mlp_recommendations[:10]

    try:
        prediction_id = save_prediction(
            xgboost_score, xgboost_category, xgboost_recommendations,
            mlp_score, mlp_category, mlp_recommendations,
            str(current_user["_id"]),
            face_image_path,
            jewelry_image_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prediction: {str(e)}")

    # Indicate that overall feedback is required
    return {
        "prediction_id": prediction_id,
        "user_id": str(current_user["_id"]),
        "prediction1": {
            "score": round(xgboost_score, 2),
            "category": xgboost_category,
            "recommendations": xgboost_recommendations,
            "feedback_required": "Overall feedback for prediction1 is required"
        },
        "prediction2": {
            "score": round(mlp_score, 2),
            "category": mlp_category,
            "recommendations": mlp_recommendations,
            "feedback_required": "Overall feedback for prediction2 is required"
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
    score: float = Form(None),  # Optional, defaults to None
    current_user: dict = Depends(get_current_user)
):
    # Score is optional for individual recommendations
    if score is not None and not (0 <= score <= 1):
        raise HTTPException(status_code=400, detail="Score must be between 0 and 1 if provided")

    if model_type not in ["prediction1", "prediction2"]:
        raise HTTPException(status_code=400, detail="Model type must be 'prediction1' or 'prediction2'")

    success = save_review(
        str(current_user["_id"]),
        prediction_id,
        model_type,
        recommendation_name,
        score if score is not None else 0.5,  # Default to 0.5 if not provided
        feedback_type="recommendation"
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return {"message": "Recommendation feedback saved successfully (optional)"}

@router.post("/feedback/prediction")
async def submit_prediction_feedback(
    prediction_id: str = Form(...),
    model_type: str = Form(...),  # "prediction1" or "prediction2"
    score: float = Form(...),  # Required
    current_user: dict = Depends(get_current_user)
):
    # Validate required score
    if not (0 <= score <= 1):
        raise HTTPException(status_code=400, detail="Score must be between 0 and 1")

    if model_type not in ["prediction1", "prediction2"]:
        raise HTTPException(status_code=400, detail="Model type must be 'prediction1' or 'prediction2'")

    success = save_review(
        str(current_user["_id"]),
        prediction_id,
        model_type,
        recommendation_name=None,
        score=score,
        feedback_type="prediction"
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return {"message": "Prediction feedback saved successfully"}