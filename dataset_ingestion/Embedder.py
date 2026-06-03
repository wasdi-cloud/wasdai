import logging
import torch

from sentence_transformers import SentenceTransformer

oLogger = logging.getLogger(__name__)

class Embedder:

    def __init__(self, sModelName):
        if not sModelName:
            raise ValueError(f"__init__: no embeddings model name provided")
        
        # Determine device and log GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        oLogger.info(f"__init__: Using device: {device}")
        if torch.cuda.is_available():
            oLogger.info(f"__init__: GPU detected: {torch.cuda.get_device_name(0)}")
        
        oLogger.info(f"__init__:  loading the embedding model: {sModelName}")
        self.model = SentenceTransformer(sModelName)
        self.model.to(device)
        oLogger.info(f"__init__: embedding model loaded on {device}")

    
    def embed(self, asTexts: list[str]) -> list[list[float]]:
        """
        Embed a list of text string
        Returns a list of float vectors.
        """
        afVectors = self.model.encode(asTexts, show_progress_bar=False, convert_to_numpy=True)
        return afVectors.tolist()