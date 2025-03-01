from fastapi import APIRouter, Depends
from api.dependencies import get_current_user
from services.database import get_db_client
from bson import ObjectId

router = APIRouter(prefix="/history", tags=["history"])

@router.get("/")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    client = get_db_client()
    db = client["jewelify"]
    predictions = list(db["recommendations"].find({"user_id": ObjectId(current_user["_id"])}).sort("timestamp", -1))
    if not predictions:
        return {"message": "No predictions found"}

    results = [
        {
            "id": str(pred["_id"]),
            "score": pred["score"],
            "category": pred["category"],
            "recommendations": pred["recommendations"],
            "timestamp": pred["timestamp"]
        }
        for pred in predictions
    ]
    return results