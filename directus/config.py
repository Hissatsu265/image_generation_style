from dotenv import load_dotenv
import os
load_dotenv()

class DirectusConfig:
    DIRECTUS_URL = os.getenv("DIRECTUS_URL")
    ACCESS_TOKEN = os.getenv("DIRECTUS_ACCESS_TOKEN")  # Replace in production
    FOLDER_ID = os.getenv("DIRECTUS_FOLDER_ID")