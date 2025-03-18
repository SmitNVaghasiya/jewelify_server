import os
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
from api.routes import auth, predictions, history
from keep_alive import start_keep_alive

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(
    title="Jewelify API",
    description="API for Jewelify application",
    version="1.0.0",
)

# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(history.router)

# Home endpoint
@app.get('/')
async def home():
    return {"Message": "Welcome to Jewelify home page"}

# Start keep-alive task on startup
@app.on_event("startup")
async def startup_event():
    start_keep_alive(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)