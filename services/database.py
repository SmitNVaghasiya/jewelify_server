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

def save_prediction(score, category, recommendations, user_id, face_data=None, jewelry_data=None):
    client = get_db_client()
    if not client:
        logger.warning("⚠️ No MongoDB client available, attempting to rebuild")
        if not rebuild_client():
            logger.error("❌ Failed to rebuild MongoDB client, cannot save prediction")
            return None

    try:
        db = client["jewelify"]
        collection = db["recommendations"]
        images_collection = db["images"]

        # Save face and jewelry images to images collection
        face_url = None
        jewelry_url = None
        if face_data:
            face_image_id = str(ObjectId())
            images_collection.insert_one({
                "_id": face_image_id,
                "name": f"face_{face_image_id}",
                "data": face_data,  # Store binary data
                "type": "face"
            })
            face_url = f"face_{face_image_id}"
        if jewelry_data:
            jewelry_image_id = str(ObjectId())
            images_collection.insert_one({
                "_id": jewelry_image_id,
                "name": f"jewelry_{jewelry_image_id}",
                "data": jewelry_data,  # Store binary data
                "type": "jewelry"
            })
            jewelry_url = f"jewelry_{jewelry_image_id}"

        prediction = {
            "user_id": ObjectId(user_id),
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "face_image": face_url,  # Reference to face image
            "jewelry_image": jewelry_url,  # Reference to jewelry image
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

        # Fetch recommendation images
        recommendations = prediction.get("recommendations", [])
        image_data = []
        for rec in recommendations:
            image_doc = images_collection.find_one({"name": rec["name"]})
            url = image_doc["url"] if image_doc and "url" in image_doc else None
            image_data.append({
                "name": rec["name"],
                "url": url,
                "score": rec.get("score", prediction.get("score", 0.0)),
                "category": rec.get("category", prediction.get("category", "Not Assigned"))
            })

        # Fetch face and jewelry images
        face_url = None
        jewelry_url = None
        if prediction.get("face_image"):
            face_doc = images_collection.find_one({"name": prediction["face_image"]})
            face_url = face_doc["url"] if face_doc and "url" in face_doc else None
        if prediction.get("jewelry_image"):
            jewelry_doc = images_collection.find_one({"name": prediction["jewelry_image"]})
            jewelry_url = jewelry_doc["url"] if jewelry_doc and "url" in jewelry_doc else None

        result = {
            "id": str(prediction["_id"]),
            "score": prediction["score"],
            "category": prediction["category"],
            "recommendations": image_data,
            "face_image": face_url,
            "jewelry_image": jewelry_url,
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
            for rec in recommendations:
                image_doc = images_collection.find_one({"name": rec["name"]})
                url = image_doc["url"] if image_doc and "url" in image_doc else None
                image_data.append({
                    "name": rec["name"],
                    "url": url,
                    "score": rec.get("score", prediction.get("score", 0.0)),
                    "category": rec.get("category", prediction.get("category", "Not Assigned"))
                })

            # Fetch face and jewelry images
            face_url = None
            jewelry_url = None
            if prediction.get("face_image"):
                face_doc = images_collection.find_one({"name": prediction["face_image"]})
                face_url = face_doc["url"] if face_doc and "url" in face_doc else None
            if prediction.get("jewelry_image"):
                jewelry_doc = images_collection.find_one({"name": prediction["jewelry_image"]})
                jewelry_url = jewelry_doc["url"] if jewelry_doc and "url" in jewelry_doc else None

            results.append({
                "id": str(prediction["_id"]),
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "face_image": face_url,
                "jewelry_image": jewelry_url,
                "timestamp": prediction["timestamp"]
            })

        logger.info(f"✅ Retrieved {len(results)} predictions for user {user_id}")
        return results
    except Exception as e:
        logger.error(f"❌ Error retrieving predictions from MongoDB: {e}")
        return {"error": f"Database error: {str(e)}"}