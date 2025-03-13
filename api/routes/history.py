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
        predictions_collection = db["recommendations"]
        reviews_collection = db["reviews"]
        predictions = list(predictions_collection.find({"user_id": ObjectId(current_user["_id"])}).sort("timestamp", -1))
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
                individual_feedback[model_type][review["recommendation_name"]] = review["review"]

        # Fetch overall prediction feedback
        prediction_reviews = list(reviews_collection.find({
            "prediction_id": ObjectId(prediction_id),
            "user_id": ObjectId(user_id),
            "feedback_type": "prediction"
        }))
        overall_feedback = {"prediction1": "Not Provided", "prediction2": "Not Provided"}
        for review in prediction_reviews:
            model_type = review["model_type"]
            if model_type in overall_feedback:
                overall_feedback[model_type] = review["review"]

        # Determine liked recommendations
        liked = {"prediction1": [], "prediction2": []}
        for model_type, rec_field in [("prediction1", "xgboost_recommendations"), ("prediction2", "fnn_recommendations")]:
            model_recs = pred.get(rec_field, [])
            model_individual_feedback = individual_feedback[model_type]
            model_overall_feedback = overall_feedback[model_type]

            for rec in model_recs:
                rec_name = rec["name"]
                # Check individual feedback first
                if rec_name in model_individual_feedback:
                    if model_individual_feedback[rec_name] == "yes":
                        liked[model_type].append(rec_name)
                # If no individual feedback, use overall feedback
                elif model_overall_feedback == "yes":
                    liked[model_type].append(rec_name)

        # Process recommendations for each model
        xgboost_recs = pred.get("xgboost_recommendations", [])
        fnn_recs = pred.get("fnn_recommendations", [])

        # Add liked status
        for rec in xgboost_recs:
            rec["liked"] = rec["name"] in liked["prediction1"]
        for rec in fnn_recs:
            rec["liked"] = rec["name"] in liked["prediction2"]

        # Sort: liked first, then by score
        xgboost_sorted = sorted(xgboost_recs, key=lambda x: (x["liked"], x["score"]), reverse=True)
        fnn_sorted = sorted(fnn_recs, key=lambda x: (x["liked"], x["score"]), reverse=True)

        # Ensure at least 10 recommendations
        liked_xgboost_count = len([r for r in xgboost_sorted if r["liked"]])
        liked_fnn_count = len([r for r in fnn_sorted if r["liked"]])
        xgboost_display = xgboost_sorted[:max(10, liked_xgboost_count)]
        fnn_display = fnn_sorted[:max(10, liked_fnn_count)]

        results.append({
            "id": prediction_id,
            "user_id": str(pred["user_id"]),
            "prediction1": {
                "score": pred["xgboost_score"],
                "category": pred["xgboost_category"],
                "recommendations": xgboost_display,
                "overall_feedback": overall_feedback["prediction1"]
            },
            "prediction2": {
                "score": pred["fnn_score"],
                "category": pred["fnn_category"],
                "recommendations": fnn_display,
                "overall_feedback": overall_feedback["prediction2"]
            },
            "face_image_path": pred.get("face_image_path"),
            "jewelry_image_path": pred.get("jewelry_image_path"),
            "timestamp": pred["timestamp"]
        })

    return results