import logging

from sentence_transformers import SentenceTransformer

oLogger = logging.getLogger(__name__)

class Embedder:

    def __init__(self, sModelName):
        if not sModelName:
            raise ValueError(f"__init__: no embeddings model name provided")
        oLogger.info(f"__init__:  loading the embedding model: {sModelName}")
        self.model = SentenceTransformer(sModelName)
        oLogger.info(f"__init__: embedding model loaded")

    
    def embed(self, asTexts: list[str]) -> list[list[float]]:
        """
        Embed a list of text string
        Returns a list of float vectors.
        """
        afVectors = self.model.encode(asTexts, show_progress_bar=False, convert_to_numpy=True)
        return afVectors.tolist()