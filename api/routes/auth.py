from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from models.user import UserRegister, UserOut
from services.auth import hash_password, create_access_token
from services.database import get_db_client
from datetime import datetime
import os

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    # Connect to the database
    try:
        client = get_db_client()
        db = client["jewelify"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database connection error: " + str(e))
    
    # Check if username or mobile number already exists
    try:
        if db["users"].find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="Username already exists")
        if db["users"].find_one({"mobileNo": user.mobileNo}):
            raise HTTPException(status_code=400, detail="Mobile number already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database query error: " + str(e))
    
    # Log received OTP (for now, OTP is not being sent via Twilio)
    print(f"Received OTP for {user.mobileNo}: {user.otp}")
    
    # ----- Twilio OTP integration (commented out) -----
    """
    from twilio.rest import Client
    try:
        twilio_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        message = twilio_client.messages.create(
            body=f"Your OTP is {user.otp}",
            from_=os.environ["TWILIO_PHONE_NUMBER"],
            to=user.mobileNo
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Twilio OTP sending failed: " + str(e))
    """
    # --------------------------------------------------

    try:
        hashed_password = hash_password(user.password)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error hashing password: " + str(e))
    
    user_data = {
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "otp": None  # Clear OTP after use
    }
    
    try:
        result = db["users"].insert_one(user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create user: " + str(e))
    
    access_token = create_access_token(data={"sub": str(result.inserted_id)})
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
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database connection error: " + str(e))
    
    try:
        user = db["users"].find_one({
            "$or": [
                {"username": form_data.username},
                {"mobileNo": form_data.username}
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database query error: " + str(e))
    
    if not user or hash_password(form_data.password) != user.get("hashed_password"):
        raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")

    try:
        access_token = create_access_token(data={"sub": str(user["_id"])})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Token creation error: " + str(e))
    
    return {"access_token": access_token, "token_type": "bearer"}
