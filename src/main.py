import json
import logging
from types import SimpleNamespace
from DatasetScanner import DatasetScanner
from ChromaStore import ChromaStore
from utils.LoggingConfiguration import setupLogging
from Embedder import Embedder

setupLogging()

from parsers.DocumentParser import DocumentParser

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

    sDatasetFolderPath = oConfig.datasetPath
    oDocumentParser = DocumentParser(sDatasetFolderPath)
    aoChunks = oDocumentParser.parseOneDocument(sFilePath)
    
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
            








def main(sFolderPath: str):

    # read the configuration file
    sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
    oConfig = readConfigFile(sConfigFilePath)


    # connect to Chroma and get info about the ingested files
    oChromaStore = ChromaStore(
        sPersistDirectory=oConfig.chromaStore.persistDirectory,
        sCollectionName=oConfig.chromaStore.collectionName
    )

    oDbSnapshot = oChromaStore.getStoredFiles()

    # understand which files are new or updated wrt what is stored in the DB
    
    # scan the file system to find the files to ingest
    sDatasetPath = oConfig.datasetPath
    oDatasetScanner = DatasetScanner(sDatasetPath)
    oDatasetSnapshot, asNew, asDeleted, asModified, asUnchanged = oDatasetScanner.findDifference(oDbSnapshot)

    # load embeddings model
    if not(asNew or asDeleted or asModified):
        oLogger.info("main. All files are updated, nothing to do")

    oEmbeddingModel = Embedder(sModelName=oConfig.embedding.modelName)

    # - delete chunks for removed files
    for sFilePath in asDeleted:
        oLogger.info(f"Deleting chunks for removed file {sFilePath}")
        oChromaStore.deleteBySourcePath(sFilePath)

    # - re-ingest modified files
    for sFilePath in asModified:
            oLogger.info(f"Re-ingesting modified file {sFilePath}")
            oChromaStore.deleteBySourcePath(sFilePath)
            ingestDocument(sFilePath, oDatasetSnapshot[sFilePath], oChromaStore, oEmbeddingModel, oConfig)
            
    # - ingest new files
    for sFilePath in asNew:
            oLogger.info(f"Ingesting new file {sFilePath}")
            ingestDocument(sFilePath, oDatasetSnapshot[sFilePath], oChromaStore, oEmbeddingModel, oConfig)  

    # provide a summary of the performed operations
    # TODO: these statistics could me more "real"
    oLogger.info(
        f"main. Pipeline complete — "
        f"ingested: {len(asNew)} new, {len(asModified)} modified, "
        f"deleted: {len(asDeleted)}, skipped: {len(asUnchanged)}"
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
    sFolderPath = "test_dataset"    # TODO: from where should I read this path? from env variable? from command line argument? 
    main(sFolderPath)