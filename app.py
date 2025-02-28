import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from pymongo import MongoClient  # Import MongoClient directly here for simplicity

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

@app.get('/')
async def home():
    return {"Message": "Welcome to Jewelify home page"}

@app.get("/get_predictions")
async def get_predictions():
    """Retrieve all predictions with image URLs"""
    try:
        # Get MongoDB URI from environment variables
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            return JSONResponse(content={"error": "MONGO_URI not found in environment variables"}, status_code=500)

        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["jewelify"]
        collection = db["recommendations"]
        images_collection = db["images"]

        # Get all predictions
        predictions = list(collection.find().sort("timestamp", -1))
        if not predictions:
            client.close()
            return JSONResponse(content={"error": "No predictions found"}, status_code=404)

        results = []
        for prediction in predictions:
            recommendations = prediction.get("recommendations", [])
            image_data = []
            for name in recommendations:
                image_doc = images_collection.find_one({"name": name})
                if image_doc:
                    image_data.append({"name": name, "url": image_doc["url"]})
                else:
                    image_data.append({"name": name, "url": None})  # Handle missing images

            results.append({
                "score": prediction.get("score"),
                "category": prediction.get("category"),
                "recommendations": image_data,
                "timestamp": prediction.get("timestamp")
            })

        client.close()  # Close the connection
        return results

    except Exception as e:
        print(f"\u274c Error retrieving predictions from MongoDB: {e}")
        return JSONResponse(content={"error": f"Database error: {str(e)}"}, status_code=500)

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT; default to 5000
    uvicorn.run(app, host="0.0.0.0", port=port)