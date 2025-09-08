import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Output directory
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Upload directory
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Log directory
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Redis configuration for job queue
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Job cleanup configuration
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", 24))  # Giữ job trong 24h
VIDEO_RETENTION_HOURS = int(os.getenv("VIDEO_RETENTION_HOURS", 72))  # Giữ video trong 72h
CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", 60))  # Cleanup mỗi 60 phút
# =======================================
SERVER_COMFYUI="127.0.0.1:8188"
WORKFLOW_INFINITETALK_PATH="/workflow/wanvideo_infinitetalk_single_example_19_8 (1).json"
# =======================================
from dotenv import load_dotenv
import os
load_dotenv()

class DirectusConfig:
    DIRECTUS_URL = os.getenv("DIRECTUS_URL")
    ACCESS_TOKEN = os.getenv("DIRECTUS_ACCESS_TOKEN")
    FOLDER_ID = os.getenv("DIRECTUS_FOLDER_ID")


# =====================================================
from datetime import datetime
from pymongo import MongoClient

# MongoDB Configuration
pass_db=os.getenv("MONGODB_PASSWORD","MONGODB_Pass")
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    f"mongodb://admin:{pass_db}@87.106.214.210:27017"
)
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "anymateme_eduhub_prod")
MONGODB_JOBS_COLLECTION = os.getenv("MONGODB_JOBS_COLLECTION", "video_jobs")
MONGODB_EFFECT_JOBS_COLLECTION = os.getenv("MONGODB_EFFECT_JOBS_COLLECTION", "effect_jobs")
MONGODB_DATABASE = "anymateme_eduhub_prod"