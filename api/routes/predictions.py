from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.collection import Collection
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from .auth import get_current_user  # Assuming this is in your project structure
from services.predictor import JewelryPredictor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# MongoDB client setup
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    logger.error("MONGO_URI not set in environment variables")
    raise ValueError("MONGO_URI is required")

client = None
try:
    client = AsyncIOMotorClient(mongo_uri)
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")

db = client["jewelry_db"] if client is not None else None
predictions_collection = db["predictions"] if db is not None else None

# Initialize predictor globally
predictor = None
try:
    predictor = JewelryPredictor()
    logger.info("JewelryPredictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize JewelryPredictor: {str(e)}")

async def save_prediction_to_db(prediction_data: dict, user_id: str) -> str:
    if predictions_collection is None:
        logger.error("Database connection not available")
        raise HTTPException(status_code=500, detail="Database connection unavailable")
    
    logger.info(f"Saving prediction to database for user {user_id}...")
    start_time = datetime.now()
    try:
        prediction_data["user_id"] = user_id
        prediction_data["timestamp"] = datetime.now().isoformat()
        result = await predictions_collection.insert_one(prediction_data)
        logger.info(f"Prediction saved to database in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    face_image_path: str = Form(...),
    jewelry_image_path: str = Form(...),
    user: dict = Depends(get_current_user),
):
    global predictor
    if predictor is None:
        try:
            predictor = JewelryPredictor()
            logger.info("JewelryPredictor reinitialized successfully")
        except Exception as e:
            logger.error(f"Failed to reinitialize JewelryPredictor: {str(e)}")
            raise HTTPException(status_code=500, detail="Predictor initialization failed")

    logger.info(f"Received prediction request for user {user['sub']}...")
    start_time = datetime.now()
    try:
        # Read and decode images
        face_contents = await face.read()
        jewelry_contents = await jewelry.read()

        face_image = cv2.imdecode(np.frombuffer(face_contents, np.uint8), cv2.IMREAD_COLOR)
        if face_image is None:
            logger.warning("Failed to decode face image")
            raise HTTPException(status_code=400, detail="Invalid or unsupported face image format")

        jewelry_image = cv2.imdecode(np.frombuffer(jewelry_contents, np.uint8), cv2.IMREAD_COLOR)
        if jewelry_image is None:
            logger.warning("Failed to decode jewelry image")
            raise HTTPException(status_code=400, detail="Invalid or unsupported jewelry image format")

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
        if not prediction_id:
            logger.error("Failed to save prediction to database")
            raise HTTPException(status_code=500, detail="Failed to save prediction")

        prediction_result["prediction_id"] = prediction_id

        logger.info(f"Prediction request completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return {
            "status": "success",
            "prediction_id": prediction_id,
            "message": "Prediction completed and saved successfully",
            "details": prediction_result
        }

    except HTTPException as e:
        logger.error(f"HTTP error during prediction: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@router.get("/get_prediction/{prediction_id}")
async def get_prediction(prediction_id: str, user: dict = Depends(get_current_user)):
    if predictions_collection is None:
        logger.error("Database connection not available")
        raise HTTPException(status_code=500, detail="Database connection unavailable")

    logger.info(f"Fetching prediction {prediction_id} for user {user['sub']}...")
    start_time = datetime.now()
    try:
        prediction = await predictions_collection.find_one({"_id": prediction_id, "user_id": user["sub"]})
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found")
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Ensure feedback scores default to 0.5 if not provided or invalid
        prediction["prediction1"] = prediction.get("prediction1", {})
        prediction["prediction2"] = prediction.get("prediction2", {})
        
        overall_feedback1 = prediction["prediction1"].get("overall_feedback")
        overall_feedback2 = prediction["prediction2"].get("overall_feedback")
        
        prediction["prediction1"]["overall_feedback"] = (
            float(overall_feedback1) if overall_feedback1 is not None and overall_feedback1 != "Not Provided" else 0.5
        )
        prediction["prediction2"]["overall_feedback"] = (
            float(overall_feedback2) if overall_feedback2 is not None and overall_feedback2 != "Not Provided" else 0.5
        )

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
    if predictions_collection is None:
        logger.error("Database connection not available")
        raise HTTPException(status_code=500, detail="Database connection unavailable")

    logger.info(f"Submitting feedback for prediction {prediction_id} for user {user['sub']}...")
    start_time = datetime.now()
    try:
        prediction = await predictions_collection.find_one({"_id": prediction_id, "user_id": user["sub"]})
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found for feedback")
            raise HTTPException(status_code=404, detail="Prediction not found")

        update_data = {}
        if feedback_type == "prediction":
            try:
                score_float = float(score)
                if not 0 <= score_float <= 1:
                    raise ValueError("Score must be between 0 and 1")
                update_data[f"{model_type}.overall_feedback"] = score_float
                update_data[f"{model_type}.feedback_required"] = False
            except ValueError as e:
                logger.warning(f"Invalid score value: {score}")
                raise HTTPException(status_code=400, detail=f"Invalid score: {str(e)}")
        elif feedback_type == "recommendation":
            if not recommendation_name:
                logger.warning("Recommendation name missing for feedback")
                raise HTTPException(status_code=400, detail="Recommendation name is required")
            try:
                score_float = float(score)
                if not 0 <= score_float <= 1:
                    raise ValueError("Score must be between 0 and 1")
                recommendations = prediction[model_type]["recommendations"]
                for rec in recommendations:
                    if rec["name"] == recommendation_name:
                        rec["feedback"] = score_float
                        break
                else:
                    logger.warning(f"Recommendation {recommendation_name} not found in {model_type}")
                    raise HTTPException(status_code=400, detail="Recommendation not found")
                update_data[f"{model_type}.recommendations"] = recommendations
            except ValueError as e:
                logger.warning(f"Invalid score value: {score}")
                raise HTTPException(status_code=400, detail=f"Invalid score: {str(e)}")
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