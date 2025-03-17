from pymongo import MongoClient
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

def get_db_client():
    global client
    if client is None:
        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas!")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            client = None
    return client

def rebuild_client():
    global client, MONGO_URI
    if not MONGO_URI:
        logger.error("Cannot rebuild client: MONGO_URI not found")
        return False
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("Successfully rebuilt MongoDB client")
        return True
    except Exception as e:
        logger.error(f"Failed to rebuild MongoDB client: {e}")
        return False

def save_prediction(prediction_data: dict, user_id: str):
    client = get_db_client()
    if not client:
        logger.warning("No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("Failed to rebuild MongoDB client, cannot save prediction")
            return None

    try:
        db = client["jewelify"]
        collection = db["recommendations"]

        prediction = {
            "user_id": ObjectId(user_id),
            "prediction1": prediction_data["prediction1"],
            "prediction2": prediction_data["prediction2"],
            "face_image_path": prediction_data["face_image_path"],
            "jewelry_image_path": prediction_data["jewelry_image_path"],
            "timestamp": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(prediction)
        logger.info(f"Saved prediction with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving prediction to MongoDB: {e}")
        return None

def get_prediction_by_id(prediction_id, user_id):
    client = get_db_client()
    if not client:
        logger.warning("No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("Failed to rebuild MongoDB client, cannot retrieve prediction")
            return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        reviews_collection = db["reviews"]

        prediction = predictions_collection.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id)
        })
        if not prediction:
            logger.warning(f"Prediction with ID {prediction_id} not found for user {user_id}")
            return {"error": "Prediction not found"}

        # Ensure prediction1 and prediction2 exist with default values
        prediction["prediction1"] = prediction.get("prediction1", {})
        prediction["prediction2"] = prediction.get("prediction2", {})

        # Fetch individual recommendation feedback
        recommendation_reviews = list(reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "recommendation"
        }))
        individual_feedback = {"prediction1": {}, "prediction2": {}}
        for review in recommendation_reviews:
            model_type = review["model_type"]
            if model_type in individual_feedback:
                individual_feedback[model_type][review["recommendation_name"]] = review["score"]

        # Fetch overall prediction feedback
        prediction_reviews = list(reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "prediction"
        }))
        overall_feedback = {"prediction1": None, "prediction2": None}
        for review in prediction_reviews:
            model_type = review["model_type"]
            if model_type in overall_feedback:
                overall_feedback[model_type] = review["score"]

        # Default to 0.5 if no feedback or invalid feedback
        overall_feedback["prediction1"] = float(overall_feedback["prediction1"]) if overall_feedback["prediction1"] is not None else 0.5
        overall_feedback["prediction2"] = float(overall_feedback["prediction2"]) if overall_feedback["prediction2"] is not None else 0.5

        # Determine liked recommendations
        liked = {"prediction1": [], "prediction2": []}
        for model_type, rec_field in [("prediction1", "prediction1"), ("prediction2", "prediction2")]:
            model_recs = prediction[model_type].get("recommendations", [])
            model_individual_feedback = individual_feedback[model_type]
            model_overall_feedback = overall_feedback[model_type]

            for rec in model_recs:
                rec_name = rec["name"]
                if rec_name in model_individual_feedback:
                    if model_individual_feedback[rec_name] >= 0.75:
                        liked[model_type].append(rec_name)
                elif model_overall_feedback >= 0.75:
                    liked[model_type].append(rec_name)

        # Add liked status to recommendations
        for model_type in ["prediction1", "prediction2"]:
            if model_type in prediction and "recommendations" in prediction[model_type]:
                for rec in prediction[model_type]["recommendations"]:
                    rec["liked"] = rec["name"] in liked[model_type]

        result = {
            "id": str(prediction["_id"]),
            "user_id": str(prediction["user_id"]),
            "prediction1": {
                "score": prediction["prediction1"].get("score", 0.0),
                "category": prediction["prediction1"].get("category", "Neutral"),
                "recommendations": prediction["prediction1"].get("recommendations", []),
                "overall_feedback": overall_feedback["prediction1"],
                "feedback_required": prediction["prediction1"].get("feedback_required", True)
            },
            "prediction2": {
                "score": prediction["prediction2"].get("score", 0.0),
                "category": prediction["prediction2"].get("category", "Neutral"),
                "recommendations": prediction["prediction2"].get("recommendations", []),
                "overall_feedback": overall_feedback["prediction2"],
                "feedback_required": prediction["prediction2"].get("feedback_required", True)
            },
            "face_image_path": prediction.get("face_image_path"),
            "jewelry_image_path": prediction.get("jewelry_image_path"),
            "timestamp": prediction["timestamp"]
        }
        logger.info(f"Retrieved prediction with ID: {prediction_id} for user {user_id}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving prediction from MongoDB: {e}")
        return {"error": f"Database error: {str(e)}"}

def save_review(user_id, prediction_id, model_type, recommendation_name, score, feedback_type):
    client = get_db_client()
    if not client:
        logger.warning("No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("Failed to rebuild MongoDB client, cannot save review")
            return False

    try:
        db = client["jewelify"]
        reviews_collection = db["reviews"]

        review_doc = {
            "user_id": ObjectId(user_id),
            "prediction_id": ObjectId(prediction_id),
            "model_type": model_type,
            "recommendation_name": recommendation_name,
            "score": float(score),
            "feedback_type": feedback_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = reviews_collection.insert_one(review_doc)
        logger.info(f"Saved review with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving review to MongoDB: {e}")
        return False