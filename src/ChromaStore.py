"""
Wrapper for ChromaDB
"""

import logging
import chromadb
from pathlib import Path
from chromadb.config import Settings
from typing import Any

oLogger = logging.getLogger(__name__)

class ChromaStore:
    
    def __init__(self, sPersistDirectory: str, sCollectionName: str):
        if not sPersistDirectory or not sCollectionName:
            oLogger.error("__init__. Missing persist directory or collection name")
            raise ValueError("Missing persist directory or collection name")

        self.client = chromadb.PersistentClient(
            path=sPersistDirectory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name = sCollectionName,
            metadata={"hnsw:space": "cosine"}
        )
        oLogger.info(f"__init__ Connected to ChromaDB collection {sCollectionName}, {self.collection.count()} chunks stored")

    
    def getStoredFiles(self) -> dict[str, str]:
        """
        Query all the stored metadata and return {filePath: fileHash}.
        """
        if self.collection.count() == 0:
            return {}
        
        oResult = self.collection.get(include=["metadatas"])
        oStoredFiles : dict[str, str] = {}
        for oMetadata in oResult["metadatas"]:
            sPath = oMetadata.get("sourcePath") # TODO: this will have to match with the name of the metadata that we give to the chunk
            sFileHash = oMetadata.get("fileHash")
            if sPath and sFileHash:
                oStoredFiles[sPath] = sFileHash
        return oStoredFiles
    
    
    def upsert(self, 
               asIds: list[str],
               afEmbeddings: list[list[float]],
               asDocuments: list[str],
               aoMetadatas: list[dict[str, Any]]):
        """
        Upsert a batch of chunks to the collection
        """
        self.collection.upsert(
            ids=asIds,
            embeddings=afEmbeddings,
            documents=asDocuments,
            metadatas=aoMetadatas
        )
        oLogger.info(f"upsert. Upserted {len(asIds)} chunks")


    def deleteBySourcePath(self, sSourcePath: str) -> bool:
        """
        Delete all the chunks belonging to a given source file
        """
        if not sSourcePath:
            oLogger.warning(f"File path {sSourcePath} does not exist")
            return False

        oResult = self.collection.get(
            where={"sourcePath": sSourcePath},
            include=[],
        )

        asIdsToDelete = oResult["ids"]

        if asIdsToDelete:
            self.collection.delete(ids=asIdsToDelete)
            oLogger.info(f"deleteBySourcePath. Deleted {len(asIdsToDelete)} chunks for {sSourcePath}")
            return True

        oLogger.warning("deleteBySourcePath. No documents to delete")
        return False