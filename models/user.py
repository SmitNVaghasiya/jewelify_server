# models/user.py
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    mobileNo: str = Field(..., min_length=10, max_length=13)
    password: str = Field(..., min_length=6)
    otp: str = Field(..., min_length=6, max_length=6)

    @field_validator("mobileNo")
    def validate_mobile(cls, v):
    # Option 1: Allow a leading plus, then digits
        import re
        if not re.match(r"^\+?\d+$", v):
            raise ValueError("Mobile number must contain only digits (optionally with a leading +)")
        return v

class UserLogin(BaseModel):
    username_or_mobile: str = Field(..., alias="username")
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