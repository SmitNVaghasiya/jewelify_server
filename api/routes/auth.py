from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from services.database import get_db_client
from services.auth import hash_password, create_access_token, verify_password
from datetime import datetime
import os
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Pydantic models
class UserRegister(BaseModel):
    username: str
    mobileNo: str
    password: str
    otp: str

class UserOut(BaseModel):
    id: str
    username: str
    mobileNo: str
    created_at: str
    access_token: str

class OtpRequest(BaseModel):
    mobileNo: str

# --- Endpoints ---

@router.get("/check-user")
async def check_user(mobile_no: str):
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
    try:
        client = get_db_client()
        db = client["jewelify"]
        otp = str(random.randint(100000, 999999))  # Generate 6-digit OTP
        
        # Log OTP for debugging (replace with actual SMS service in production)
        logger.info(f"Generated OTP for {request.mobileNo}: {otp}")

        # Store OTP in the database
        db["users"].update_one(
            {"mobileNo": request.mobileNo},
            {"$set": {"otp": otp, "otp_created_at": datetime.utcnow().isoformat()}},
            upsert=True
        )

        # Twilio integration (uncomment and configure if using Twilio)
        """
        from twilio.rest import Client
        twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        message = twilio_client.messages.create(
            body=f"Your OTP for Jewelify is {otp}",
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            to=request.mobileNo
        )
        logger.info(f"Twilio SMS sent: {message.sid}")
        """

        return {"message": "OTP sent successfully"}
    except Exception as e:
        logger.error(f"Error sending OTP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send OTP: {str(e)}")

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    try:
        client = get_db_client()
        db = client["jewelify"]
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    # Check if username or mobile number already exists
    if db["users"].find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    if db["users"].find_one({"mobileNo": user.mobileNo}):
        raise HTTPException(status_code=400, detail="Mobile number already exists")

    # Verify OTP
    user_temp = db["users"].find_one({"mobileNo": user.mobileNo})
    if not user_temp or user_temp.get("otp") != user.otp:
        logger.warning(f"Invalid OTP for {user.mobileNo}: {user.otp}")
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Save user data
    user_data = {
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "otp": None  # Clear OTP after verification
    }
    
    try:
        result = db["users"].insert_one(user_data)
        access_token = create_access_token(data={"sub": str(result.inserted_id)})
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
    
    # Clear OTP from database
    db["users"].update_one({"mobileNo": user.mobileNo}, {"$set": {"otp": None}})
    
    return {
        "id": str(result.inserted_id),
        "username": user.username,
        "mobileNo": user.mobileNo,
        "created_at": user_data["created_at"],
        "access_token": access_token
    }

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
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