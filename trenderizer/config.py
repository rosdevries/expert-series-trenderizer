import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VOCAB_DIR = BASE_DIR / "vocab"
LOCAL_CACHE_DIR = BASE_DIR / "local_cache"

ON24_CLIENT_ID = os.getenv("ON24_CLIENT_ID", "48920")
ON24_TOKEN_KEY = os.getenv("ON24_TOKEN_KEY")
ON24_TOKEN_SECRET = os.getenv("ON24_TOKEN_SECRET")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Tags that qualify an event for inclusion in the Trenderizer
TARGET_TAGS: frozenset[str] = frozenset({
    "Expert Series Webinars",
    "Lunch and Learn",
    "Lunch & Learn",
    "Customer Expert Series",
})

PAGE_SIZE = 100
REQUEST_TIMEOUT = 60
BACKFILL_DAYS = 365
DEFAULT_FETCH_DAYS = 30
