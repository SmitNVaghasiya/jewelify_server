from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.collection import Collection
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import logging
import os
from .auth import get_current_user
from services.predictor import JewelryPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# MongoDB client
client = AsyncIOMotorClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client["jewelry_db"]
predictions_collection: Collection = db["predictions"]

# Initialize predictor
predictor = JewelryPredictor()

async def save_prediction_to_db(prediction_data: dict, user_id: str) -> str:
    logger.info("Saving prediction to database...")
    start_time = datetime.now()
    prediction_data["user_id"] = user_id
    prediction_data["timestamp"] = datetime.now()
    result = await predictions_collection.insert_one(prediction_data)
    logger.info(f"Prediction saved to database in {(datetime.now() - start_time).total_seconds():.2f} seconds")
    return str(result.inserted_id)

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),
    jewelry_image_path: str = Form(...),
    user: dict = Depends(get_current_user),
):
    logger.info("Received prediction request...")
    start_time = datetime.now()
    try:
        # Read and decode images
        face_contents = await face.read()
        jewelry_contents = await jewelry.read()

        face_image = cv2.imdecode(np.frombuffer(face_contents, np.uint8), cv2.IMREAD_COLOR)
        jewelry_image = cv2.imdecode(np.frombuffer(jewelry_contents, np.uint8), cv2.IMREAD_COLOR)

        # Validate images
        is_valid_face, face_message = predictor.validate_face_image(face_image)
        if not is_valid_face:
            logger.warning(f"Face image validation failed: {face_message}")
            raise HTTPException(status_code=400, detail=f"Uploaded face image is invalid: {face_message}")

        is_valid_jewelry, jewelry_message = predictor.validate_image(jewelry_image)
        if not is_valid_jewelry:
            logger.warning(f"Jewelry image validation failed: {jewelry_message}")
            raise HTTPException(status_code=400, detail=f"Uploaded jewelry image is invalid: {jewelry_message}")

        # Run predictions
        prediction_result = await predictor.predict_both(
            face_image,
            jewelry_image,
            face_image_path,
            jewelry_image_path,
        )

        # Save to database
        prediction_id = await save_prediction_to_db(prediction_result, user["sub"])
        prediction_result["prediction_id"] = prediction_id

        logger.info(f"Prediction request completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return prediction_result

    except HTTPException as e:
        logger.error(f"HTTP error during prediction: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@router.get("/get_prediction/{prediction_id}")
async def get_prediction(prediction_id: str, user: dict = Depends(get_current_user)):
    logger.info(f"Fetching prediction {prediction_id}...")
    start_time = datetime.now()
    try:
        prediction = await predictions_collection.find_one({"_id": prediction_id, "user_id": user["sub"]})
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found")
            raise HTTPException(status_code=404, detail="Prediction not found")
        prediction["prediction_id"] = prediction_id
        logger.info(f"Prediction {prediction_id} fetched in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return prediction
    except Exception as e:
        logger.error(f"Error fetching prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching prediction: {str(e)}")

@router.post("/feedback/{feedback_type}")
async def submit_feedback(
    feedback_type: str,
    prediction_id: str = Form(...),
    model_type: str = Form(...),
    recommendation_name: Optional[str] = Form(None),
    score: str = Form(...),
    user: dict = Depends(get_current_user),
):
    logger.info(f"Submitting feedback for prediction {prediction_id}...")
    start_time = datetime.now()
    try:
        prediction = await predictions_collection.find_one({"_id": prediction_id, "user_id": user["sub"]})
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found for feedback")
            raise HTTPException(status_code=404, detail="Prediction not found")

        update_data = {}
        if feedback_type == "prediction":
            update_data[f"{model_type}.overall_feedback"] = score
            update_data[f"{model_type}.feedback_required"] = False
        elif feedback_type == "recommendation":
            if not recommendation_name:
                logger.warning("Recommendation name missing for feedback")
                raise HTTPException(status_code=400, detail="Recommendation name is required")
            recommendations = prediction[model_type]["recommendations"]
            for rec in recommendations:
                if rec["name"] == recommendation_name:
                    rec["feedback"] = score
                    break
            update_data[f"{model_type}.recommendations"] = recommendations
        else:
            logger.warning(f"Invalid feedback type: {feedback_type}")
            raise HTTPException(status_code=400, detail="Invalid feedback type")

        await predictions_collection.update_one(
            {"_id": prediction_id}, {"$set": update_data}
        )
        logger.info(f"Feedback for prediction {prediction_id} submitted in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return {"message": "Feedback submitted successfully"}
    except HTTPException as e:
        logger.error(f"HTTP error during feedback submission: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")