import asyncio
import aiohttp
import logging
from fastapi import FastAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Free API to ping (e.g., public API health check)
KEEP_ALIVE_URL = "https://api.publicapis.org/health"

async def keep_alive_task(app: FastAPI):
    """Background task to ping a URL every 13 minutes to keep the Render instance alive."""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(KEEP_ALIVE_URL) as response:
                    if response.status == 200:
                        logger.info("Keep-alive ping successful")
                    else:
                        logger.warning(f"Keep-alive ping failed with status: {response.status}")
        except Exception as e:
            logger.error(f"Error in keep-alive ping: {e}")
        # Wait for 14 minutes (850 seconds)
        await asyncio.sleep(850)

def start_keep_alive(app: FastAPI):
    """Start the keep-alive task."""
    loop = asyncio.get_event_loop()
    loop.create_task(keep_alive_task(app))