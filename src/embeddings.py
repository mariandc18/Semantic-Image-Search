import open_clip as clip
import torch
from PIL import Image
import numpy as np
from config import CLIP_MODEL

class CLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = clip.create_model_and_transforms(
    CLIP_MODEL, 
    pretrained="openai",
    device=self.device
)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        
        return image_features.cpu().numpy()[0]
    
    def embed_text(self, text: str) -> np.ndarray:
        text_input = self.model.tokenizer.encode(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features.cpu().numpy()[0]