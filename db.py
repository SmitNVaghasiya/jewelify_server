import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_mongo_client():
    """Return a MongoDB client if connection is successful"""
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        print("🚨 MONGO_URI not found in environment variables")
        return None

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')  # Test connection
        print("✅ Successfully connected to MongoDB Atlas!")
        return client
    except Exception as e:
        print(f"🚨 Failed to connect to MongoDB Atlas: {e}")
        return None

def save_prediction(score, category, recommendations, client=None):
    """Save prediction to MongoDB and return the inserted _id as a string"""
    if not client:
        client = get_mongo_client()
        if not client:
            return None

    try:
        db = client["jewelify"]
        collection = db["recommendations"]
        prediction = {
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(prediction)
        return str(result.inserted_id)  # Return the _id as a string
    except Exception as e:
        print(f"❌ Error saving prediction to MongoDB: {e}")
        return None
    finally:
        if client:
            client.close()

def get_latest_prediction(client=None):
    """Retrieve the latest prediction with image URLs"""
    if not client:
        client = get_mongo_client()
        if not client:
            return None

    try:
        db = client["jewelify"]  # Corrected to match database name
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        # Get the most recent prediction
        prediction = predictions_collection.find_one(sort=[("timestamp", -1)])
        if not prediction:
            return {"error": "No predictions found"}

        # Fetch image URLs for recommendations
        recommendations = prediction.get("recommendations", [])
        image_data = []
        for name in recommendations:
            image_doc = images_collection.find_one({"name": name})
            if image_doc:
                image_data.append({"name": name, "url": image_doc["url"]})
            else:
                image_data.append({"name": name, "url": None})  # Handle missing images

        result = {
            "score": prediction["score"],
            "category": prediction["category"],
            "recommendations": image_data,
            "timestamp": prediction["timestamp"]
        }
        return result
    except Exception as e:
        print(f"❌ Error retrieving prediction from MongoDB: {e}")
        return None
    finally:
        if client:
            client.close()