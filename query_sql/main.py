from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from core.db import init_db
from utils.utils import cleanup_old_audio_files
from api.routes import register_routes

# Import register_websocket with error handling to debug
try:
    from api.websocket import register_websocket
    logging.info("Successfully imported register_websocket from api.websocket")
except ImportError as e:
    logging.error(f"Failed to import register_websocket from api.websocket: {str(e)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="QuerySQL Backend",
    description="A FastAPI backend for document processing and querying.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes and WebSocket endpoints
try:
    register_routes(app)
    logger.info("Routes registered successfully")
except Exception as e:
    logger.error(f"Failed to register routes: {str(e)}")
    raise

try:
    register_websocket(app)
    logger.info("WebSocket endpoints registered successfully")
except Exception as e:
    logger.error(f"Failed to register WebSocket endpoints: {str(e)}")
    raise

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    try:
        cleanup_old_audio_files()
        logger.info("Old audio files cleaned up successfully")
    except Exception as e:
        logger.warning(f"Failed to clean up old audio files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start Uvicorn server: {str(e)}")
        raise