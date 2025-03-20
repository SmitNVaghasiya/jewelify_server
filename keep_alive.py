# keep_alive.py
import asyncio
import aiohttp
import logging
from fastapi import FastAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the app's own health check endpoint to keep the server alive
# Replace with your Render URL or use an environment variable
KEEP_ALIVE_URL = "http://localhost:8000/health"
# Interval for keep-alive pings (14 minutes = 840 seconds)
KEEP_ALIVE_INTERVAL = 840
# Number of retries for failed pings
RETRY_ATTEMPTS = 3
# Delay between retries
RETRY_DELAY = 30  # 30 seconds
# Timeout for HTTP requests
REQUEST_TIMEOUT = 10  # 10 seconds

async def keep_alive_task(app: FastAPI):
    """Background task to ping a URL periodically to keep the Render instance alive."""
    while True:
        attempt = 0
        success = False
        while attempt < RETRY_ATTEMPTS and not success:
            try:
                async with aiohttp.ClientSession() as session:
                    # Set a timeout for the HTTP request
                    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                    async with session.get(KEEP_ALIVE_URL, timeout=timeout) as response:
                        if response.status == 200:
                            logger.info(f"Keep-alive ping successful to {KEEP_ALIVE_URL}")
                            success = True
                        else:
                            logger.warning(
                                f"Keep-alive ping failed with status: {response.status} (Attempt {attempt + 1}/{RETRY_ATTEMPTS})"
                            )
                            attempt += 1
                            if attempt < RETRY_ATTEMPTS:
                                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                                await asyncio.sleep(RETRY_DELAY)
            except asyncio.TimeoutError:
                logger.error(
                    f"Keep-alive ping timed out after {REQUEST_TIMEOUT} seconds (Attempt {attempt + 1}/{RETRY_ATTEMPTS})"
                )
                attempt += 1
                if attempt < RETRY_ATTEMPTS:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(
                    f"Error in keep-alive ping: {e} (Attempt {attempt + 1}/{RETRY_ATTEMPTS})"
                )
                attempt += 1
                if attempt < RETRY_ATTEMPTS:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)

        if not success:
            logger.critical(
                f"Keep-alive ping failed after {RETRY_ATTEMPTS} attempts. This may indicate a network or server issue. Will retry after {KEEP_ALIVE_INTERVAL} seconds."
            )
            # Optionally, add a notification mechanism here (e.g., send an email or log to a monitoring service)

        # Wait for the configured interval before the next ping
        await asyncio.sleep(KEEP_ALIVE_INTERVAL)

def start_keep_alive(app: FastAPI):
    """Start the keep-alive task."""
    try:
        loop = asyncio.get_event_loop()
        # Ensure the loop is running
        if loop.is_running():
            loop.create_task(keep_alive_task(app))
        else:
            loop.run_until_complete(keep_alive_task(app))
        logger.info("Keep-alive task started successfully")
    except Exception as e:
        logger.error(f"Failed to start keep-alive task: {e}")