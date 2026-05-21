import logging

from abc import ABC, abstractmethod
from pathlib import Path
from llama_index.core.schema import BaseNode, Document

oLogger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for all file parsers"""
    
    def __init__(self, iChunkSize: int = 1000, iChunkOverlap: int = 100):
        self.chunkSize = iChunkSize
        self.chunkOverlap = iChunkOverlap
    
    @abstractmethod
    def parse(self, oFilePath: Path, bDebugContent: bool = False) -> list[dict]:
        """Parse the document and return a list of chunk dictionaries."""
        pass

    def _getChunksFromBaseNodes(self, aoBaseNodes: list[BaseNode]) -> list[dict]:
        """
        Given a list of BaseNodes representing the sections of a document,
        split them into chunks and return the list of chunks, each of them being a dictionary with keys: text and metadata.
        """
        if not aoBaseNodes:
            return []
        
        aoChunks = [
            {
                "text": oNode.get_content(),
                "metadata": oNode.metadata,  
            }
            for oNode in aoBaseNodes
            if oNode.get_content().strip()
        ]

        return aoChunks
