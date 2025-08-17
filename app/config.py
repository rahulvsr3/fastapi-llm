import os
from dotenv import load_dotenv

load_dotenv()

# Local folder to cache the model
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
CODESPACE_STORAGE_LIMIT_GB = 10  # Free tier limit
