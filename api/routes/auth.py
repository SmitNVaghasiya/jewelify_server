from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from services.database import get_db_client  # Using MongoDB client
from services.auth import hash_password, create_access_token, verify_password
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create APIRouter with prefix and tags
router = APIRouter(prefix="/auth", tags=["auth"])

# Pydantic models
class UserRegister(BaseModel):
    username: str
    mobileNo: str
    password: str
    id: str  # Firebase UID

class UserOut(BaseModel):
    id: str
    username: str
    mobileNo: str
    created_at: str
    access_token: str

class OtpRequest(BaseModel):
    mobileNo: str

# --- Endpoints ---

@router.get("/check-user/{mobile_no}")
async def check_user(mobile_no: str):
    """Check if a user exists by mobile number using MongoDB."""
    try:
        client = get_db_client()
        db = client["jewelify"]
        user = db["users"].find_one({"mobileNo": mobile_no})
        return {"exists": bool(user)}
    except Exception as e:
        logger.error(f"Error checking user: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/send-otp")
async def send_otp(request: OtpRequest):
    """Placeholder endpoint for sending OTP (handled by Firebase, not FastAPI)."""
    try:
        logger.info(f"Request to send OTP for {request.mobileNo} - Handled by Firebase")
        return {"message": "OTP sending initiated (handled by Firebase)"}
    except Exception as e:
        logger.error(f"Error processing OTP request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process OTP request: {str(e)}")

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    try:
        client = get_db_client()
        db = client["jewelify"]
    except Exception as e:
        logger.error(f"Database connection error: $e")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    # Check if username or mobile number already exists
    if db["users"].find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    if db["users"].find_one({"mobileNo": user.mobileNo}):
        raise HTTPException(status_code=400, detail="Mobile number already exists")
    if db["users"].find_one({"firebase_uid": user.id}):
        raise HTTPException(status_code=400, detail="Firebase UID already registered")

    # Hash password
    hashed_password = hash_password(user.password)

    # Store the Firebase UID separately and let MongoDB generate a new ObjectId
    user_data = {
        "firebase_uid": user.id,  # Store Firebase UID separately
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        result = db["users"].insert_one(user_data)
        user_id = str(result.inserted_id)  # MongoDB-generated ObjectId
        access_token = create_access_token(data={"sub": user_id})
    except Exception as e:
        logger.error(f"Error registering user: $e")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    return {
        "id": user_id,
        "username": user.username,
        "mobileNo": user.mobileNo,
        "created_at": user_data["created_at"],
        "access_token": access_token
    }

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login a user using username/mobileNo and password, using MongoDB."""
    try:
        client = get_db_client()
        db = client["jewelify"]
        user = db["users"].find_one({
            "$or": [
                {"username": form_data.username},
                {"mobileNo": form_data.username}
            ]
        })
    except Exception as e:
        logger.error(f"Database error during login: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}