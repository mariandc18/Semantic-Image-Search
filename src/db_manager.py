import chromadb
from chromadb.config import Settings
from pathlib import Path
from config import DB_DIR, COLLECTION_NAME

class ChromaManager:
    def __init__(self):
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(DB_DIR),
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_image(self, image_id: str, embedding: list, image_path: str):
        self.collection.add(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[{"image_path": image_path}]
        )
    
    def search_by_embedding(self, embedding: list, n_results: int = 5):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results
    
    def get_all_images(self):
        return self.collection.get()
    
    def clear_collection(self):
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )