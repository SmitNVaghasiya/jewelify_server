# # api/routes/auth.py
# from fastapi import APIRouter, Depends, HTTPException
# from fastapi.security import OAuth2PasswordRequestForm
# from models.user import UserRegister, UserOut
# from services.auth import hash_password, create_access_token
# from services.database import get_db_client
# from datetime import datetime
# import random
# import string

# router = APIRouter(prefix="/auth", tags=["auth"])

# def generate_otp(length=6):
#     """Generate a random OTP of specified length."""
#     return ''.join(random.choices(string.digits, k=length))

# @router.post("/send_otp")
# async def send_otp(mobileNo: str):
#     client = get_db_client()
#     db = client["jewelify"]
    
#     # Check if the mobile number already exists
#     if db["users"].find_one({"mobileNo": mobileNo}):
#         raise HTTPException(status_code=400, detail="Mobile number already registered")
    
#     # Generate OTP
#     otp = generate_otp()
    
#     # Store OTP temporarily in the database (or use a cache like Redis in production)
#     # For simplicity, we'll store it in the users collection with a temporary flag
#     db["users"].update_one(
#         {"mobileNo": mobileNo},
#         {
#             "$set": {
#                 "mobileNo": mobileNo,
#                 "otp": otp,
#                 "otp_timestamp": datetime.utcnow().isoformat(),
#                 "temporary": True  # Mark as temporary until registration is complete
#             }
#         },
#         upsert=True
#     )
    
#     # In a real app, send the OTP via SMS (e.g., using Twilio)
#     # For now, we'll just print it to the console for testing
#     print(f"OTP for {mobileNo}: {otp}")
#     # Example with Twilio (uncomment and configure if you have Twilio credentials):
#     # from twilio.rest import Client
#     # twilio_client = Client("your_account_sid", "your_auth_token")
#     # twilio_client.messages.create(
#     #     body=f"Your OTP is {otp}",
#     #     from_="your_twilio_number",
#     #     to=mobileNo
#     # )
    
#     return {"message": f"OTP sent to {mobileNo}"}

# @router.post("/register", response_model=UserOut)
# async def register(user: UserRegister):
#     client = get_db_client()
#     db = client["jewelify"]
    
#     # Check if username or mobile number already exists (excluding temporary entries)
#     if db["users"].find_one({"username": user.username, "temporary": {"$ne": True}}):
#         raise HTTPException(status_code=400, detail="Username already exists")
#     if db["users"].find_one({"mobileNo": user.mobileNo, "temporary": {"$ne": True}}):
#         raise HTTPException(status_code=400, detail="Mobile number already exists")

#     # Verify OTP
#     temp_user = db["users"].find_one({"mobileNo": user.mobileNo, "temporary": True})
#     if not temp_user or temp_user["otp"] != user.otp:
#         raise HTTPException(status_code=400, detail="Invalid OTP")
    
#     # OTP is valid, proceed with registration
#     hashed_password = hash_password(user.password)
#     user_data = {
#         "username": user.username,
#         "mobileNo": user.mobileNo,
#         "hashed_password": hashed_password,
#         "created_at": datetime.utcnow().isoformat(),
#         "otp": None,
#         "temporary": False  # Mark as a completed registration
#     }
    
#     # Update the temporary entry to a permanent one
#     db["users"].update_one(
#         {"mobileNo": user.mobileNo, "temporary": True},
#         {"$set": user_data}
#     )
    
#     return {
#         "id": str(temp_user["_id"]),
#         "username": user.username,
#         "mobileNo": user.mobileNo,
#         "created_at": user_data["created_at"]
#     }

# @router.post("/login")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     client = get_db_client()
#     db = client["jewelify"]
#     user = db["users"].find_one({"$or": [{"username": form_data.username}, {"mobileNo": form_data.username}]})
#     if not user or not hash_password(form_data.password) == user["hashed_password"]:
#         raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")

#     access_token = create_access_token(data={"sub": str(user["_id"])})
#     return {"access_token": access_token, "token_type": "bearer"}

# api/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from models.user import UserRegister, UserOut
from services.auth import hash_password, create_access_token
from services.database import get_db_client
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut)
async def register(user: UserRegister):
    client = get_db_client()
    db = client["jewelify"]
    
    # Check if username or mobile number already exists
    if db["users"].find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    if db["users"].find_one({"mobileNo": user.mobileNo}):
        raise HTTPException(status_code=400, detail="Mobile number already exists")

    # Since Firebase verifies the OTP, we can skip OTP validation in the backend
    # Optionally, you can log the OTP for auditing purposes
    print(f"Received OTP for {user.mobileNo}: {user.otp}")
    
    # Proceed with registration
    hashed_password = hash_password(user.password)
    user_data = {
        "username": user.username,
        "mobileNo": user.mobileNo,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "otp": None  # Clear OTP after use (optional, since Firebase verified it)
    }
    
    result = db["users"].insert_one(user_data)
    
    return {
        "id": str(result.inserted_id),
        "username": user.username,
        "mobileNo": user.mobileNo,
        "created_at": user_data["created_at"]
    }

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    client = get_db_client()
    db = client["jewelify"]
    user = db["users"].find_one({"$or": [{"username": form_data.username}, {"mobileNo": form_data.username}]})
    if not user or not hash_password(form_data.password) == user["hashed_password"]:
        raise HTTPException(status_code=400, detail="Incorrect username/mobileNo or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}