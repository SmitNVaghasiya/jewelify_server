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

def save_prediction(xgboost_score, xgboost_category, xgboost_recommendations, mlp_score, mlp_category, mlp_recommendations, user_id, face_image_path, jewelry_image_path):
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
            "xgboost_score": xgboost_score,
            "xgboost_category": xgboost_category,
            "xgboost_recommendations": xgboost_recommendations,
            "mlp_score": mlp_score,
            "mlp_category": mlp_category,
            "mlp_recommendations": mlp_recommendations,
            "face_image_path": face_image_path,
            "jewelry_image_path": jewelry_image_path,
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

        # Determine liked recommendations
        liked = {"prediction1": [], "prediction2": []}
        for model_type, rec_field in [("prediction1", "xgboost_recommendations"), ("prediction2", "mlp_recommendations")]:
            model_recs = prediction.get(rec_field, [])
            model_individual_feedback = individual_feedback[model_type]
            model_overall_feedback = overall_feedback[model_type]

            for rec in model_recs:
                rec_name = rec["name"]
                if rec_name in model_individual_feedback:
                    if model_individual_feedback[rec_name] >= 0.75:
                        liked[model_type].append(rec_name)
                elif model_overall_feedback is not None and model_overall_feedback >= 0.75:
                    liked[model_type].append(rec_name)

        # Add liked status to recommendations
        for model_type, rec_field in [("prediction1", "xgboost_recommendations"), ("prediction2", "mlp_recommendations")]:
            if rec_field in prediction:
                for rec in prediction[rec_field]:
                    rec["liked"] = rec["name"] in liked[model_type]

        result = {
            "id": str(prediction["_id"]),
            "user_id": str(prediction["user_id"]),
            "prediction1": {
                "score": prediction["xgboost_score"],
                "category": prediction["xgboost_category"],
                "recommendations": prediction["xgboost_recommendations"],
                "overall_feedback": overall_feedback["prediction1"] if overall_feedback["prediction1"] is not None else "Not Provided"
            },
            "prediction2": {
                "score": prediction["mlp_score"],
                "category": prediction["mlp_category"],
                "recommendations": prediction["mlp_recommendations"],
                "overall_feedback": overall_feedback["prediction2"] if overall_feedback["prediction2"] is not None else "Not Provided"
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