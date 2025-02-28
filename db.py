import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["jewelify"]
predictions_collection = db["recommendations"]
images_collection = db["images"]

def save_prediction(score, category, recommendations):
    """Save prediction to MongoDB"""
    try:
        prediction = {
            "score": score,
            "category": category,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = predictions_collection.insert_one(prediction)
        return str(result.inserted_id)  # Return the inserted ID
    except Exception as e:
        print(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_latest_prediction():
    """Retrieve the latest prediction with image URLs"""
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