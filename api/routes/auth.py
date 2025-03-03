from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from pymongo import MongoClient
from bson.objectid import ObjectId
import jwt
import datetime
from typing import Optional
from models.user import UserOut, UserRegister
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI router
router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

# Database connection (replace with your MongoDB URI)
def get_db_client():
    return MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI

# Secret key for JWT (replace with your actual secret)
SECRET_KEY = "your-secret-key"  # Replace with a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    try:
        client = get_db_client()
        db = client["jewelify"]
        
        # Check if user exists by mobileNo
        if db["users"].find_one({"mobileNo": user.mobileNo}):
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password (simplified for this example; use bcrypt or similar in production)
        hashed_password = user.password  # In practice, hash with bcrypt: hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
        
        # Insert user into MongoDB
        user_data = {
            "username": user.username,
            "mobileNo": user.mobileNo,
            "password": hashed_password,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        result = db["users"].insert_one(user_data)
        
        # Generate JWT token
        access_token = jwt.encode(
            {"sub": str(result.inserted_id), "exp": datetime.datetime.utcnow().timestamp() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        # Return the user object with the token
        response = {
            "id": str(result.inserted_id),
            "username": user.username,
            "mobileNo": user.mobileNo,
            "created_at": user_data["created_at"],
            "access_token": access_token
        }
        
        logger.info(f"User registered successfully: {response}")
        return response
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        client = get_db_client()
        db = client["jewelify"]
        
        # Find user by username or mobileNo
        user = db["users"].find_one({"$or": [{"username": form_data.username}, {"mobileNo": form_data.username}]})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password (simplified; use bcrypt in production)
        if user["password"] != form_data.password:  # In practice, use: bcrypt.checkpw(form_data.password.encode(), user["password"].encode())
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate JWT token
        access_token = jwt.encode(
            {"sub": str(user["_id"]), "exp": datetime.datetime.utcnow().timestamp() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        # Return login response (matching UserOut for consistency)
        response = {
            "id": str(user["_id"]),
            "username": user["username"],
            "mobileNo": user["mobileNo"],
            "created_at": user["created_at"],
            "access_token": access_token
        }
        
        logger.info(f"User logged in successfully: {response}")
        return response
    except Exception as e:
        logger.error(f"Error logging in user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to login: {str(e)}")

@router.get("/check-user/{mobileNo}")
async def check_user_exists(mobileNo: str):
    try:
        client = get_db_client()
        db = client["jewelify"]
        
        exists = bool(db["users"].find_one({"mobileNo": mobileNo}))
        logger.info(f"Checked if user exists for mobileNo {mobileNo}: {exists}")
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Error checking user existence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check user: {str(e)}")