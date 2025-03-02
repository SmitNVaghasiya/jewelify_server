import os
import logging
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = None

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

def save_prediction(score, category, recommendations, user_id=None):
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
            "score": score,
            "category": category,
            "recommendations": recommendations,  # Expected as list of strings or dicts
            "timestamp": datetime.utcnow().isoformat()
        }
        if user_id:
            try:
                prediction["user_id"] = ObjectId(user_id)
            except Exception as e:
                logger.error(f"Invalid user_id format: {e}")
                return None

        result = collection.insert_one(prediction)
        logger.info(f"✅ Saved prediction with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_prediction_by_id(prediction_id):
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

        try:
            prediction = predictions_collection.find_one({"_id": ObjectId(prediction_id)})
        except Exception as e:
            logger.error(f"Invalid prediction_id format: {e}")
            return {"error": "Invalid prediction ID format"}

        if not prediction:
            logger.warning(f"⚠️ Prediction with ID {prediction_id} not found")
            return {"error": "Prediction not found"}

        recommendations = prediction.get("recommendations", [])
        image_data = []
        for item in recommendations:
            if isinstance(item, str):
                image_doc = images_collection.find_one({"name": item})
                image_data.append({
                    "name": item,
                    "url": image_doc["url"] if image_doc and "url" in image_doc else None
                })
            elif isinstance(item, dict) and "name" in item:
                image_data.append({
                    "name": item["name"],
                    "url": item.get("url", None)
                })
            else:
                image_data.append({"name": str(item), "url": None})

        result = {
            "id": str(prediction["_id"]),
            "score": prediction["score"],
            "category": prediction["category"],
            "recommendations": image_data,
            "timestamp": prediction["timestamp"]
        }

        logger.info(f"✅ Retrieved prediction with ID: {prediction_id}")
        return result
    except Exception as e:
        logger.error(f"❌ Error retrieving prediction from MongoDB: {e}")
        return {"error": str(e)}

def get_all_predictions():
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

        predictions = list(predictions_collection.find().sort("timestamp", -1))
        if not predictions:
            logger.warning("⚠️ No predictions found")
            return {"error": "No predictions found"}

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for item in recommendations:
                if isinstance(item, str):
                    image_doc = images_collection.find_one({"name": item})
                    image_data.append({
                        "name": item,
                        "url": image_doc["url"] if image_doc and "url" in image_doc else None
                    })
                elif isinstance(item, dict) and "name" in item:
                    image_data.append({
                        "name": item["name"],
                        "url": item.get("url", None)
                    })
                else:
                    image_data.append({"name": str(item), "url": None})

            results.append({
                "id": str(prediction["_id"]),
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "timestamp": prediction["timestamp"]
            })

        logger.info(f"✅ Retrieved {len(results)} predictions")
        return results
    except Exception as e:
        logger.error(f"❌ Error retrieving predictions from MongoDB: {e}")
        return {"error": str(e)}