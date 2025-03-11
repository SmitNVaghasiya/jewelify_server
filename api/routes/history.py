from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_current_user
from services.database import get_db_client
from bson import ObjectId

router = APIRouter(prefix="/history", tags=["history"])

@router.get("/")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    client = get_db_client()
    if not client:
        return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_cursor = db["recommendations"].find({"user_id": ObjectId(current_user["_id"])})
        predictions = list(predictions_cursor.sort("timestamp", -1))
    except Exception as e:
        return {"error": "Database error: " + str(e)}

    if not predictions:
        return {"message": "No predictions found", "recommendations": []}

    results = []
    for pred in predictions:
        recommendations = pred.get("recommendations", [])
        formatted_recommendations = []
        for item in recommendations:
            if isinstance(item, str):
                formatted_recommendations.append({
                    "name": item,
                    "url": None,
                    "score": pred.get("score", 0.0),
                    "category": pred.get("category", "Not Assigned")
                })
            elif isinstance(item, dict) and "name" in item:
                formatted_recommendations.append({
                    "name": item["name"],
                    "url": item.get("url", None),
                    "score": item.get("score", pred.get("score", 0.0)),
                    "category": item.get("category", pred.get("category", "Not Assigned"))
                })
            else:
                formatted_recommendations.append({
                    "name": str(item),
                    "url": None,
                    "score": pred.get("score", 0.0),
                    "category": pred.get("category", "Not Assigned")
                })

        results.append({
            "id": str(pred["_id"]),
            "user_id": str(pred["user_id"]),
            "score": pred["score"],
            "category": pred["category"],
            "recommendations": formatted_recommendations,
            "face_image_path": pred.get("face_image"),  # Return local path
            "jewelry_image_path": pred.get("jewelry_image"),  # Return local path
            "timestamp": pred["timestamp"]
        })

    return results