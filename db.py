import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import ssl
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

# MongoDB connection with SSL/TLS configuration
MONGO_URI = os.getenv("MONGO_URI")
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

try:
    client = MongoClient(MONGO_URI, ssl=ssl_context, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')  # Test connection
    print("✅ Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"🚨 Failed to connect to MongoDB Atlas: {e}")
    client = None

if client:
    db = client["jewelify_db"]
    predictions_collection = db["recommendations"]
    images_collection = db["images"]
else:
    db = None
    predictions_collection = None
    images_collection = None

def save_prediction(score, category, recommendations):
    """Save prediction to MongoDB and return the inserted _id as a string"""
    if not predictions_collection:
        print("❌ MongoDB client not initialized")
        return None

    try:
        prediction = {
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = predictions_collection.insert_one(prediction)
        return str(result.inserted_id)  # Return the _id as a string
    except Exception as e:
        print(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_latest_prediction():
    """Retrieve the latest prediction with image URLs"""
    if not predictions_collection:
        print("❌ MongoDB client not initialized")
        return None

    try:
        # Get the most recent prediction
        prediction = predictions_collection.find().sort("timestamp", -1).limit(1)[0]
        if not prediction:
            return None

        # Fetch image URLs for recommendations
        recommendations = prediction["recommendations"]
        image_data = []
        for name in recommendations:
            image_doc = images_collection.find_one({"name": name})
            if image_doc:
                image_data.append({"name": name, "url": image_doc["url"]})
            else:
                image_data.append({"name": name, "url": None})  # Handle missing images

        return {
            "score": prediction["score"],
            "category": prediction["category"],
            "recommendations": image_data,
            "timestamp": prediction["timestamp"]
        }
    except Exception as e:
        print(f"❌ Error retrieving prediction from MongoDB: {e}")
        return None