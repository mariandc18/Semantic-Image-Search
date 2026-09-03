from pathlib import Path
from src.embeddings import CLIPEmbedder
from src.db_manager import ChromaManager
from config import IMAGES_DIR, TOP_K_RESULTS

class ImageSearch:
    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.db = ChromaManager()
    
    def index_images(self, image_dir: Path = IMAGES_DIR):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = [f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions]
        
        for idx, image_path in enumerate(images):
            embedding = self.embedder.embed_image(str(image_path))
            self.db.add_image(
                image_id=f"img_{idx}",
                embedding=embedding.tolist(),
                image_path=str(image_path)
            )
    
    def search_by_text(self, query: str, top_k: int = TOP_K_RESULTS):
        embedding = self.embedder.embed_text(query)
        results = self.db.search_by_embedding(embedding.tolist(), n_results=top_k)
        return self._format_results(results)
    
    def search_by_image(self, image_path: str, top_k: int = TOP_K_RESULTS):
        embedding = self.embedder.embed_image(image_path)
        results = self.db.search_by_embedding(embedding.tolist(), n_results=top_k)
        return self._format_results(results)
    
    def _format_results(self, results):
        formatted = []
        if results['ids'] and len(results['ids']) > 0:
            for idx, image_id in enumerate(results['ids'][0]):
                formatted.append({
                    'id': image_id,
                    'path': results['metadatas'][0][idx]['image_path'],
                    'distance': results['distances'][0][idx] if results['distances'] else 0
                })
        return formatted