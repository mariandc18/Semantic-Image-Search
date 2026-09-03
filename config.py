import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "chroma_db"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MODEL = "ViT-B/32"  
COLLECTION_NAME = "image_similarity"

TOP_K_RESULTS = 5  