import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global MongoDB client
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("🚨 MONGO_URI not found in environment variables")
    client = None
else:
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')  # Test connection
        print("✅ Successfully connected to MongoDB Atlas!")
    except Exception as e:
        print(f"🚨 Failed to connect to MongoDB Atlas: {e}")
        client = None

def save_prediction(score, category, recommendations):
    """Save prediction to MongoDB and return the inserted _id as a string"""
    if not client:
        print("❌ No MongoDB client available")
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
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error saving prediction to MongoDB: {e}")
        return None

def get_all_predictions():
    """Retrieve all predictions with image URLs"""
    if not client:
        return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        images_collection = db["images"]

        # Get all predictions
        predictions = list(predictions_collection.find().sort("timestamp", -1))
        if not predictions:
            return {"error": "No predictions found"}

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for name in recommendations:
                image_doc = images_collection.find_one({"name": name})
                if image_doc:
                    image_data.append({"name": name, "url": image_doc["url"]})
                else:
                    image_data.append({"name": name, "url": None})

            results.append({
                "score": prediction["score"],
                "category": prediction["category"],
                "recommendations": image_data,
                "timestamp": prediction["timestamp"]
            })

        return results
    except Exception as e:
        print(f"❌ Error retrieving predictions from MongoDB: {e}")
        return {"error": str(e)}