from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import get_current_user
from services.database import get_db_client
from bson import ObjectId

router = APIRouter(prefix="/history", tags=["history"])

@router.get("/")
async def get_user_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    client = get_db_client()
    if not client:
        return {"error": "Database connection error"}

    try:
        db = client["jewelify"]
        predictions_collection = db["recommendations"]
        reviews_collection = db["reviews"]
        predictions = list(predictions_collection.find({"user_id": ObjectId(current_user["_id"])}).sort("timestamp", -1).skip(skip).limit(limit))
    except Exception as e:
        return {"error": "Database error: " + str(e)}

    if not predictions:
        return {"message": "No predictions found", "history": []}

    results = []
    for pred in predictions:
        prediction_id = str(pred["_id"])
        user_id = str(current_user["_id"])

        # Fetch individual recommendation feedback
        recommendation_reviews = list(reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "recommendation"
        }))
        individual_feedback = {"prediction1": {}, "prediction2": {}}
        for review in recommendation_reviews:
            model_type = review["model_type"]
            if model_type in individual_feedback:
                individual_feedback[model_type][review["recommendation_name"]] = review["score"]

        # Fetch overall prediction feedback
        prediction_reviews = list(reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "prediction"
        }))
        overall_feedback = {"prediction1": None, "prediction2": None}
        for review in prediction_reviews:
            model_type = review["model_type"]
            if model_type in overall_feedback:
                overall_feedback[model_type] = review["score"]

        # Process recommendations for each model
        xgboost_recs = pred.get("xgboost_recommendations", [])
        mlp_recs = pred.get("mlp_recommendations", [])

        # Apply feedback scores to recommendations
        for rec in xgboost_recs:
            rec_name = rec["name"]
            # Default liked status based on model prediction score
            rec["liked"] = rec["score"] >= 75.0
            # Apply user score: individual feedback takes precedence, else use overall feedback
            if rec_name in individual_feedback["prediction1"]:
                rec["user_score"] = individual_feedback["prediction1"][rec_name]
                rec["liked"] = rec["user_score"] >= 0.75
            elif overall_feedback["prediction1"] is not None:
                rec["user_score"] = overall_feedback["prediction1"]
                rec["liked"] = rec["user_score"] >= 0.75
            else:
                rec["user_score"] = None  # Will trigger feedback requirement

        for rec in mlp_recs:
            rec_name = rec["name"]
            # Default liked status based on model prediction score
            rec["liked"] = rec["score"] >= 75.0
            # Apply user score: individual feedback takes precedence, else use overall feedback
            if rec_name in individual_feedback["prediction2"]:
                rec["user_score"] = individual_feedback["prediction2"][rec_name]
                rec["liked"] = rec["user_score"] >= 0.75
            elif overall_feedback["prediction2"] is not None:
                rec["user_score"] = overall_feedback["prediction2"]
                rec["liked"] = rec["user_score"] >= 0.75
            else:
                rec["user_score"] = None  # Will trigger feedback requirement

        # Sort: liked first, then by score
        xgboost_sorted = sorted(xgboost_recs, key=lambda x: (x["liked"], x["score"]), reverse=True)
        mlp_sorted = sorted(mlp_recs, key=lambda x: (x["liked"], x["score"]), reverse=True)

        # Enforce exactly 10 recommendations per model
        xgboost_display = xgboost_sorted[:10]
        mlp_display = mlp_sorted[:10]

        results.append({
            "id": prediction_id,
            "user_id": str(pred["user_id"]),
            "prediction1": {
                "score": pred["xgboost_score"],
                "category": pred["xgboost_category"],
                "recommendations": xgboost_display,
                "overall_feedback": overall_feedback["prediction1"] if overall_feedback["prediction1"] is not None else "Not Provided",
                "feedback_required": "Overall feedback for prediction1 is required" if overall_feedback["prediction1"] is None else None
            },
            "prediction2": {
                "score": pred["mlp_score"],
                "category": pred["mlp_category"],
                "recommendations": mlp_display,
                "overall_feedback": overall_feedback["prediction2"] if overall_feedback["prediction2"] is not None else "Not Provided",
                "feedback_required": "Overall feedback for prediction2 is required" if overall_feedback["prediction2"] is None else None
            },
            "face_image_path": pred.get("face_image_path"),
            "jewelry_image_path": pred.get("jewelry_image_path"),
            "timestamp": pred["timestamp"]
        })

    return results