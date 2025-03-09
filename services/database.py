from pymongo import MongoClient
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

def get_db_client():
    global client
    if client is None:
        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            logger.info("✅ Successfully connected to MongoDB Atlas!")
        except Exception as e:
            logger.error(f"🚨 Failed to connect to MongoDB Atlas: {e}")
            client = None
    return client

def rebuild_client():
    global client, MONGO_URI
    if not MONGO_URI:
        logger.error("🚨 Cannot rebuild client: MONGO_URI not found")
        return False
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("✅ Successfully rebuilt MongoDB client")
        return True
    except Exception as e:
        logger.error(f"🚨 Failed to rebuild MongoDB client: {e}")
        return False

def save_prediction(score, category, recommendations, user_id):
    client = get_db_client()
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot save prediction")
            return None

    try:
        db = client["jewelify"]
        collection = db["recommendations"]
        prediction = {
            "user_id": ObjectId(user_id),
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(prediction)
        logger.info(f"✅ Saved prediction with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_prediction_by_id(prediction_id, user_id):
    client = get_db_client()
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot retrieve prediction")
            return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        prediction = predictions_collection.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id)
        })
        if not prediction:
            logger.warning(f"⚠️ Prediction with ID {prediction_id} not found for user {user_id}")
            return {"error": "Prediction not found"}

        recommendations = prediction.get("recommendations", [])
        image_data = []
        for name in recommendations:
            image_doc = images_collection.find_one({"name": name})
            url = None
            if image_doc and "url" in image_doc:
                url = image_doc["url"]  # Use the URL directly from the database
            image_data.append({
                "name": name,
                "url": url  # No fallback URL; rely on S3 URL
            })

        result = {
            "id": str(prediction["_id"]),
            "score": prediction["score"],
            "category": prediction["category"],
            "recommendations": image_data,
            "timestamp": prediction["timestamp"]
        }
        logger.info(f"✅ Retrieved prediction with ID: {prediction_id} for user {user_id}")
        return result
    except Exception as e:
        logger.error(f"❌ Error retrieving prediction from MongoDB: {e}")
        return {"error": f"Database error: {str(e)}"}

def get_user_predictions(user_id):
    client = get_db_client()
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot retrieve predictions")
            return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        predictions = list(predictions_collection.find({"user_id": ObjectId(user_id)}).sort("timestamp", -1))
        if not predictions:
            logger.warning(f"⚠️ No predictions found for user {user_id}")
            return {"error": "No predictions found"}

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for name in recommendations:
                image_doc = images_collection.find_one({"name": name})
                url = None
                if image_doc and "url" in image_doc:
                    url = image_doc["url"]  # Use the URL directly from the database
                image_data.append({
                    "name": name,
                    "url": url  # No fallback URL; rely on S3 URL
                })

            results.append({
                "id": str(prediction["_id"]),
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "timestamp": prediction["timestamp"]
            })

        logger.info(f"✅ Retrieved {len(results)} predictions for user {user_id}")
        return results
    except Exception as e:
        logger.error(f"❌ Error retrieving predictions from MongoDB: {e}")
        return {"error": f"Database error: {str(e)}"}