import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import uvicorn
from predictor import get_predictor, predict_compatibility
from db import save_prediction, get_prediction_by_id, get_all_predictions
from pymongo import MongoClient
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from bson import ObjectId
import logging

# Load environment variables
load_dotenv()

# Define paths and constants
MODEL_PATH = os.getenv("MODEL_PATH", "rl_jewelry_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
PAIRWISE_FEATURES_PATH = os.getenv("PAIRWISE_FEATURES_PATH", "pairwise_features.npy")
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize predictor globally
predictor = get_predictor(MODEL_PATH, SCALER_PATH, PAIRWISE_FEATURES_PATH)

# FastAPI app
app = FastAPI()

# MongoDB client
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    logging.info("✅ Connected to MongoDB Atlas!")
except Exception as e:
    logging.error(f"🚨 Failed to connect to MongoDB: {e}")
    client = None

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models for User
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    mobileNo: str = Field(..., min_length=10, max_length=15)
    password: str = Field(..., min_length=6)
    otp: str = Field(..., min_length=4, max_length=6)

    @validator("mobileNo")
    def validate_mobile(cls, v):
        if not v.isdigit():
            raise ValueError("Mobile number must contain only digits")
        return v

class UserLogin(BaseModel):
    username_or_mobile: str = Field(..., alias="username")  # Alias to match OAuth2 form
    password: str = Field(...)

class UserOut(BaseModel):
    id: str
    username: str
    mobileNo: str
    created_at: str

    class Config:
        from_attributes = True
        populate_by_name = True
        json_encoders = {ObjectId: str}

# Token dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db = client["jewelify"]
    user = db["users"].find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
    return user

# Helper to create JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Home endpoint
@app.get('/')
async def home():
    return {"Message": "Welcome to Jewelify home page"}

# Registration endpoint
@app.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    db = client["jewelify"]
    if db["users"].find_one({"username": user.username}) or db["users"].find_one({"mobileNo": user.mobileNo}):
        raise HTTPException(status_code=400, detail="Username or mobile number already exists")
    
    # Simulate OTP verification (replace with real OTP service)
    expected_otp = "123456"  # Placeholder
    if user.otp != expected_otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    hashed_password = pwd_context.hash(user.password)
    user_data = {
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "otp": None  # Clear OTP after verification
    }
    result = db["users"].insert_one(user_data)
    return {
        "id": str(result.inserted_id),
        "username": user.username,
        "mobileNo": user.mobileNo,
        "created_at": user_data["created_at"]
    }

# Login endpoint
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = client["jewelify"]
    user = db["users"].find_one({"$or": [{"username": form_data.username}, {"mobileNo": form_data.username}]})
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": str(user["_id"])}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# Predict endpoint (unchanged except for dependency)
@app.post("/predict")
async def predict(
    face: UploadFile = File(...),
    jewelry: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    global predictor
    if predictor is None:
        predictor = get_predictor(MODEL_PATH, SCALER_PATH, PAIRWISE_FEATURES_PATH)
        if predictor is None:
            raise HTTPException(status_code=500, detail="Model is not loaded properly")

    if not face.content_type.startswith('image/') or not jewelry.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")

    try:
        face_data = await face.read()
        jewelry_data = await jewelry.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded images")

    score, category, recommendations = predict_compatibility(predictor, face_data, jewelry_data)
    if score is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    prediction_id = save_prediction(score, category, recommendations, str(current_user["_id"]))
    return {
        "prediction_id": prediction_id,
        "score": score,
        "category": category,
        "recommendations": recommendations,
    }

# History endpoint (unchanged for brevity, already uses user_id)
@app.get("/history")
async def get_user_history(current_user: dict = Depends(get_current_user)):
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

# Existing endpoints (unchanged for brevity)
@app.get("/get_prediction/{prediction_id}")
async def get_prediction(prediction_id: str):
    result = get_prediction_by_id(prediction_id)
    if "error" in result:
        status_code = 404 if result["error"] == "Prediction not found" else 500
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result

@app.get("/get_predictions")
async def get_predictions():
    result = get_all_predictions()
    if "error" in result:
        status_code = 500 if result["error"] != "No predictions found" else 404
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)