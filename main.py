import os
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
from api.routes import auth, predictions, history

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(history.router)

# Home endpoint
@app.get('/')
async def home():
    return {"Message": "Welcome to Jewelify home page"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)