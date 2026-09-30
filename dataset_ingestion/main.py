import json
import logging
from typing import Any, Dict, List, Tuple
import chromadb
from types import SimpleNamespace

from llama_index.core import Document
from dataset_ingestion.DatasetScanner import DatasetScanner
from dataset_ingestion.ChromaStore import ChromaStore
from utils.LoggingConfiguration import setupLogging
from dataset_ingestion.Embedder import Embedder
from dataset_ingestion.parsers.DocumentParserFactory import DocumentParserFactory
from collections import Counter
from pathlib import Path
from utils.WasdiConfig import WasdiConfig
from langchain_openai import ChatOpenAI
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

setupLogging()

# from dataset_ingestion.parsers.DocumentParser import DocumentParser

oLogger = logging.getLogger(__name__)


def readConfigFile(sConfigFilePath):
    """
    Reads the configuration file and returns it as an object
    """
    with open(sConfigFilePath, "r") as oConfigFile:
        sConfigContent = oConfigFile.read()
    # Get the config as an object
    oConfig = json.loads(sConfigContent, object_hook=lambda d: SimpleNamespace(**d))
    oConfig.myFilePath = sConfigFilePath
    return oConfig


# FILE INGESTION STEP
def ingestDocument(
        sFilePath: str,
        sFileHash: str,
        oStore: ChromaStore,
        oEmbedder: Embedder,
        oConfig: dict
):
    """
    #TODO: ADD DESCRIPTION
    # ideally here we want to:
    # - parse a file
    # - generate the chungs
    # - generate the embeddings for the chunks
    # - understand if the file is already in the vector store (by hash) and update it if needed
    """

    if not sFilePath or not sFileHash:
        oLogger.warning("ingestDocument. Missing file path or hash code")
        raise ValueError("Not enough information provided to ingest document")

    oLogger.info(f"ingestDocument. Processing file {sFilePath} with hash {sFileHash}")

    # get the proper parser for the file type
    oParser = DocumentParserFactory.getParser(sFilePath)

    if oParser is None:
        oLogger.warning(f"ingestDocument. No parser found for file {sFilePath}. Skipping.")
        return

    aoChunks = oParser.parse(sFilePath, bDebugContent=True)
    
    if not aoChunks:
         oLogger.warning(f"ingestDocument. No chunks produced for file {sFilePath}. Skipping")
         return
    
    asIds = [f"{sFileHash}_chunk_{i}" for i in range(len(aoChunks))]
    aoMetadata = [{
            "sourcePath": sFilePath,
            "fileHash": sFileHash,
            "chunkIndex": i,
            "category": oChunk["metadata"].get("category", "Unknown"),
            "pageNumber": oChunk["metadata"].get("page_number", None)
        }
        for i, oChunk in enumerate(aoChunks)
    ]

    asTexts = [oChunk["text"] for oChunk in aoChunks]

    afEmbeddings = oEmbedder.embed(asTexts)

    oStore.upsert(asIds=asIds,
                  afEmbeddings=afEmbeddings,
                  asDocuments=asTexts,
                  aoMetadatas=aoMetadata)

    oLogger.info(f"ingestDocument. Ingested {len(aoChunks)} chunks from {sFilePath}")


def visualizeDbContent():
    """
    Utility function to visualize the content of the ChromaDB collection
    """
    client = chromadb.PersistentClient("C:\\WASDI\\ChromaDB")
    collections = client.list_collections()
    print("Available Collections:")
    if not collections:
        print("No collections found in this directory.")
    else:
        for col in collections:
            # Each 'col' is a Collection object
            print(f" - {col.name}")
    collection = client.get_collection(name="embeddings")
    print(f"Number of elements in the collection: {collection.count()}")
    # Peek at the first 5 items
    results = collection.peek(limit=100)

    print(f"{'ID':<20} | {'Category':<15} | {'Snippet'}")
    print("-" * 80)

    for i in range(len(results["ids"])):
        sDocId = results["ids"][i]
        sContent = results["documents"][i][:100] # Just the first 200 chars
        sCategory = results["metadatas"][i].get("category", "N/A")
        sFileName = results["metadatas"][i].get("sourcePath", "N/A")
        print(f"{sFileName} | {sDocId} | {sCategory} | {sContent}...")


def visualizeDbContentByExtension():
    """
    Utility function to visualize how many unique files are stored per file extension
    """
    from collections import defaultdict
    import os

    client = chromadb.PersistentClient("C:\\WASDI\\ChromaDB")
    collection = client.get_collection(name="embeddings")

    results = collection.get(include=["metadatas"])

    unique_paths = set()
    for metadata in results["metadatas"]:
        source_path = metadata.get("sourcePath", "")
        if source_path:
            unique_paths.add(source_path)

    extension_counts = defaultdict(int)
    for path in unique_paths:
        _, ext = os.path.splitext(path)
        ext = ext.lower() if ext else "(no extension)"
        extension_counts[ext] += 1

    total_unique_files = len(unique_paths)
    print(f"Files per extension (unique files: {total_unique_files}):")
    print("-" * 40)

    for ext, count in sorted(extension_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext:<20} : {count}")


def testFiltering(
    sUserQuery: str = "platform architecture details",
) -> List[Document]:
    """
    Directly tests metadata filtering on the Chroma vector store 
    without triggering the LLM chain.
    """
    
    s_sConfigFile = "C:\\WASDI\\GIT\\wasdai\\config_new.json"

    if not (s_oConfig := WasdiConfig(s_sConfigFile)):
        logging.error("Failed to load configuration")
        raise RuntimeError(f"Could not load config from {s_sConfigFile}")

    # Load Embeddings
    s_oEmbeddingConfig = getattr(s_oConfig, "embedding", None)
    s_sEmbeddingModelName = getattr(s_oEmbeddingConfig, "modelName", "BAAI/bge-m3")
    s_sHuggingFaceToken = getattr(s_oEmbeddingConfig, "huggingface_token", "")

    aoEmbeddingArgs: Dict[str, Any] = {
        "model_name": s_sEmbeddingModelName
    }

    if s_sHuggingFaceToken:
        os.environ["HF_TOKEN"] = s_sHuggingFaceToken
        aoEmbeddingArgs["model_kwargs"] = {"token": s_sHuggingFaceToken}

    if not (s_oEmbeddings := HuggingFaceEmbeddings(**aoEmbeddingArgs)):
        logging.error("Failed to load embeddings")
        raise RuntimeError("Could not load embeddings")

    # Load Vector Store
    s_sPersistDirectory = s_oConfig.chromaStore.persistDirectory
    s_oVectorStore = Chroma(
        collection_name="embeddings",
        embedding_function=s_oEmbeddings,
        persist_directory=s_sPersistDirectory
    )

    if not s_oVectorStore:
        logging.error("Failed to load vector store")
        raise RuntimeError(f"Could not load vector store from {s_sPersistDirectory}")

    # Metadata Filter Specification
    oMetadataFilter = {
        "$and": [
            {"source_type": {"$eq": "user_doc"}},
            {"component": {"$eq": "eo_app"}}
        ]
    }

    logging.info(f"Executing direct similarity search with filter: {oMetadataFilter}")

    # Query vector store directly bypassing LLM chain
    aoResultsWithScores: List[Tuple[Document, float]] = s_oVectorStore.similarity_search_with_score(
        query=sUserQuery,
        k=10,
        filter=oMetadataFilter
    )

    if not aoResultsWithScores:
        logging.warning("No documents returned matching the specified query and filter criteria.")
        return []

    # Validate returned metadata fields
    logging.info(f"Retrieved {len(aoResultsWithScores)} documents matching filters:")
    
    aoFilteredDocuments: List[Document] = []
    
    for iIdx, (oDoc, fScore) in enumerate(aoResultsWithScores, start=1):
        sSourceType = oDoc.metadata.get("source_type")
        sComponent = oDoc.metadata.get("component")
        sPageContent = oDoc.page_content
        
        logging.info(
            f"\n--- [Document {iIdx}] ---"
            f"\nScore: {fScore:.4f}"
            f"\nsource_type: {sSourceType} | component: {sComponent}"
            f"\nText Chunk Content:\n{sPageContent}"
            f"\n{'-' * 40}"
        )
        
        # Hard assertions to fail test if Chroma filter leaks incorrect metadata
        assert sSourceType == "user_doc", f"Expected 'user_doc', got '{sSourceType}'"
        assert sComponent == "eo_app", f"Expected 'platform', got '{sComponent}'"
        
        aoFilteredDocuments.append(oDoc)

    return aoFilteredDocuments


def main():

    # read the configuration file
    sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
    oConfig = readConfigFile(sConfigFilePath)


    # connect to Chroma and get info about the ingested files
    oChromaStore = ChromaStore(
        sPersistDirectory=oConfig.chromaStore.persistDirectory,
        sCollectionName=oConfig.chromaStore.collectionName
    )

    oDbSnapshot = oChromaStore.getStoredFiles()

    oLogger.info(f"main. Number iles currently stored in the Chroma vector store: {len(oDbSnapshot.items())}")
    for sFilePath, sFileHash in oDbSnapshot.items():
        oLogger.debug(f"main. File in DB: {sFilePath}, hash: {sFileHash}")

    # understand which files are new or updated wrt what is stored in the DB
    
    # scan the file system to find the files to ingest
    asDatasetPath = oConfig.datasetPaths
    oDatasetScanner = DatasetScanner(asDatasetPath)
    oDatasetSnapshot, asNew, asDeleted, asModified, asUnchanged = oDatasetScanner.findDifference(oDbSnapshot)

    if not(asNew or asDeleted or asModified):
        oLogger.info("main. All files are updated, nothing to do")
        return
    
    oLogger.info(f"main. Dataset scan")

    # - delete chunks for removed files
    oLogger.info(f"\t* Deleted files")
    if not asDeleted:
        oLogger.info(f"\t\t No deleted files")
    for sFilePath in asDeleted:
        oLogger.info(f"\t\t  {sFilePath}")
        oChromaStore.deleteBySourcePath(sFilePath)

    oEmbeddingModel = Embedder(sModelName=oConfig.embedding.modelName)
    
    # - re-ingest modified files
    oLogger.info(f"\t* Modified files")
    if not asModified:
        oLogger.info(f"\t\t No modified files")
    for sFilePath in asModified:
        oLogger.info(f"\t\t  {sFilePath} (will be re-ingested)")
        oChromaStore.deleteBySourcePath(sFilePath)
        ingestDocument(sFilePath, oDatasetSnapshot[sFilePath], oChromaStore, oEmbeddingModel, oConfig)

    # - ingest new files
    oLogger.info(f"\t* New files")
    if not asNew:
        oLogger.info(f"\t\t No new files")
    for sFilePath in asNew:
        oLogger.info(f"\t\t  {sFilePath}")
        ingestDocument(sFilePath, oDatasetSnapshot[sFilePath], oChromaStore, oEmbeddingModel, oConfig) 

    oLogger.info(f"\t* Unchanged files")
    for sFilePath in asUnchanged:
        oLogger.info(f"\t\t  {sFilePath}")


    # provide a summary of the performed operations
    # TODO: these statistics could me more "real"
    oLogger.info(
        f"main. Pipeline complete — "
        f"ingested: {len(asNew)} new, {len(asModified)} modified, "
        f"deleted: {len(asDeleted)},"
        f"skipped: {len(asUnchanged)}"
    )


    """

    oDocumentParser = DocumentParser(sFolderPath)
    asChunks = oDocumentParser.parseOneDocument(
        sFilePath="C:\\WASDI\\GIT\\wasdai\\test_dataset\\Tutorial_eDrift_v05.docx.pdf", 
        bDebugContent=True)
    for sChunk in asChunks:
        oLogger.info(f"main. Parsed chunk:\n{sChunk}\n--- END CHUNK ---")
    # oLogger.info(f"main. Parsed content:\n{sContent}")
    # oLogger.info(f"main. Parsed content:\n{sContent}")
    """

if __name__ == "__main__":

    iParameter = 4

    if iParameter == 1:
        main()
    elif iParameter == 2:
        visualizeDbContent()
    elif iParameter == 3:
        visualizeDbContentByExtension()
    elif iParameter == 4:
        testFiltering("flood archive")
