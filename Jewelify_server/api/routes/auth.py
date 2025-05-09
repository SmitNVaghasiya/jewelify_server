from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from services.database import get_db_client
from services.auth import hash_password, create_access_token, verify_password, generate_otp, send_otp_via_sms, store_otp, verify_otp
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from api.dependencies import get_current_user

# Load environment variables
load_dotenv()

# Configure logging based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

class UserRegister(BaseModel):
    username: str
    mobileNo: str
    password: str

class UserOut(BaseModel):
    id: str
    username: str
    mobileNo: str
    created_at: str
    access_token: str = None

class OtpRequest(BaseModel):
    mobileNo: str

class OtpVerify(BaseModel):
    mobileNo: str
    otp: str

@router.get("/check-user/{mobile_no}")
async def check_user(mobile_no: str):
    """Check if a user exists by mobile number using MongoDB."""
    logger.info(f"Checking if user with mobile number {mobile_no} exists")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = db["users"].find_one({"mobileNo": mobile_no})
        exists = bool(user)
        logger.debug(f"User with mobile number {mobile_no} exists: {exists}")
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Error checking user with mobile number {mobile_no}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/send-otp")
async def send_otp(request: OtpRequest):
    """Generate and send OTP to the user's mobile number via SMS."""
    mobile_no = request.mobileNo
    logger.info(f"Request to send OTP to {mobile_no}")
    try:
        # Validate mobile number format
        if not mobile_no.startswith('+') or len(mobile_no) < 10:
            logger.warning(f"Invalid mobile number format: {mobile_no}")
            raise HTTPException(status_code=400, detail="Invalid mobile number format. Use +[country code][number]")
        
        # Generate OTP
        otp = generate_otp()
        # Send OTP via SMS
        if not send_otp_via_sms(mobile_no, otp):
            logger.error(f"Failed to send OTP to {mobile_no}")
            raise HTTPException(status_code=500, detail="Failed to send OTP via SMS")
        # Store OTP in the database
        if not store_otp(mobile_no, otp):
            logger.error(f"Failed to store OTP for {mobile_no}")
            raise HTTPException(status_code=500, detail="Failed to store OTP")
        logger.info(f"OTP sent to {mobile_no}")
        return {"message": f"OTP sent to {mobile_no}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error sending OTP to {mobile_no}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process OTP request: {str(e)}")

@router.post("/verify-otp")
async def verify_otp_endpoint(request: OtpVerify):
    """Verify the OTP provided by the user."""
    mobile_no = request.mobileNo
    otp = request.otp
    logger.info(f"Verifying OTP for {mobile_no}")
    try:
        if not mobile_no.startswith('+') or len(mobile_no) < 10:
            logger.warning(f"Invalid mobile number format: {mobile_no}")
            raise HTTPException(status_code=400, detail="Invalid mobile number format. Use +[country code][number]")
        if not verify_otp(mobile_no, otp):
            logger.warning(f"Invalid or expired OTP for {mobile_no}")
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        logger.info(f"OTP verified successfully for {mobile_no}")
        return {"message": "OTP verified successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error verifying OTP for {mobile_no}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify OTP: {str(e)}")

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    """Register a new user."""
    logger.info(f"Registering new user with username {user.username} and mobile number {user.mobileNo}")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    # Check for existing username or mobile number
    if db["users"].find_one({"username": user.username}):
        logger.warning(f"Username {user.username} already exists")
        raise HTTPException(status_code=400, detail="Username already exists")
    if db["users"].find_one({"mobileNo": user.mobileNo}):
        logger.warning(f"Mobile number {user.mobileNo} already exists")
        raise HTTPException(status_code=400, detail="Mobile number already exists")

    hashed_password = hash_password(user.password)
    user_data = {
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        result = db["users"].insert_one(user_data)
        user_id = str(result.inserted_id)
        access_token = create_access_token(data={"sub": user_id})
    except Exception as e:
        logger.error(f"Error registering user {user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

    logger.info(f"User {user_id} registered successfully")
    return {
        "id": user_id,
        "username": user.username,
        "mobileNo": user.mobileNo,
        "created_at": user_data["created_at"],
        "access_token": access_token,
    }

@router.post("/login", response_model=UserOut)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login a user using username/mobileNo and password."""
    logger.info(f"Login attempt for {form_data.username}")
    try:
        client = get_db_client()
        if not client:
            logger.error("Database connection not available")
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        db = client["jewelify"]
        user = db["users"].find_one({
            "$or": [
                {"username": form_data.username},
                {"mobileNo": form_data.username}
            ]
        })
    except Exception as e:
        logger.error(f"Database error during login for {form_data.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        logger.warning(f"Failed login attempt for {form_data.username}: Incorrect username/mobileNo or password")
        raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    logger.info(f"User {user['_id']} logged in successfully")
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "mobileNo": user["mobileNo"],
        "created_at": user["created_at"],
        "access_token": access_token,
    }