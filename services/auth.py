import os
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random
import string
from twilio.rest import Client
from services.database import get_db_client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 43200))

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# OTP settings
OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 5

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_otp(length: int = OTP_LENGTH) -> str:
    """Generate a random OTP of specified length."""
    digits = string.digits
    return ''.join(random.choice(digits) for _ in range(length))

def send_otp_via_sms(mobile_no: str, otp: str) -> bool:
    """Send OTP to the given mobile number via SMS using Twilio."""
    try:
        message = twilio_client.messages.create(
            body=f"Your OTP for Jewelify is {otp}. It is valid for {OTP_EXPIRY_MINUTES} minutes.",
            from_=TWILIO_PHONE_NUMBER,
            to=mobile_no
        )
        logger.info(f"OTP sent to {mobile_no}: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP to {mobile_no}: {e}")
        return False

def store_otp(mobile_no: str, otp: str) -> bool:
    """Store the OTP in MongoDB with an expiration time."""
    client = get_db_client()
    if not client:
        logger.error("Cannot store OTP: No MongoDB client available")
        return False

    try:
        db = client["jewelify"]
        otps_collection = db["otps"]
        expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        otp_doc = {
            "mobileNo": mobile_no,
            "otp": otp,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expiry.isoformat()
        }
        otps_collection.insert_one(otp_doc)
        logger.info(f"OTP stored for {mobile_no}")
        return True
    except Exception as e:
        logger.error(f"Error storing OTP for {mobile_no}: {e}")
        return False

def verify_otp(mobile_no: str, otp: str) -> bool:
    """Verify the OTP for the given mobile number."""
    client = get_db_client()
    if not client:
        logger.error("Cannot verify OTP: No MongoDB client available")
        return False

    try:
        db = client["jewelify"]
        otps_collection = db["otps"]
        otp_doc = otps_collection.find_one(
            {"mobileNo": mobile_no, "otp": otp},
            sort=[("created_at", -1)]  # Get the most recent OTP
        )
        if not otp_doc:
            logger.warning(f"No OTP found for {mobile_no} or OTP does not match")
            return False

        expiry = datetime.fromisoformat(otp_doc["expires_at"])
        if datetime.utcnow() > expiry:
            logger.warning(f"OTP for {mobile_no} has expired")
            return False

        # OTP is valid, delete it from the database
        otps_collection.delete_one({"_id": otp_doc["_id"]})
        logger.info(f"OTP verified for {mobile_no}")
        return True
    except Exception as e:
        logger.error(f"Error verifying OTP for {mobile_no}: {e}")
        return False