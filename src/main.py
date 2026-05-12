import json
import logging
from types import SimpleNamespace
from DatasetScanner import DatasetScanner
from ChromaStore import ChromaStore
from utils.LoggingConfiguration import setupLogging

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
        # ChromaStore --> maybe I want it as a singleton. Not sure I want this
):
    """
    #TODO: ADD DESCRIPTION
    # ideally here we want to:
    # - parse a file
    # - generate the chungs
    # - generate the embeddings for the chunks
    # - understand if the file is already in the vector store (by hash) and update it if needed
    """




def main(sFolderPath: str):

    # read the configuration file
    sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
    oConfig = readConfigFile(sConfigFilePath)

    # scan the file system to find the files to ingest
    sDatasetPath = oConfig.datasetPath
    oDatasetScanner = DatasetScanner(sDatasetPath)
    oPathHashDict = oDatasetScanner.scan()          # get a dictionay {path:sha256}

    """
    for k, v in oPathHashDict.items():
        print(k + " " + v)
    """

    # connect to Chroma and get the known files --> this I will need to know what it means and what Chroma returns when I want the "known files"
    oChromaStore = ChromaStore(
        sPersistDirectory=oConfig.chromaStore.persistDirectory,
        sCollectionName=oConfig.chromaStore.collectionName
    )

    oChromaStore.getStoredFiles()

    # understand which files are new or updated by comparing the known files with the files found in the file system (by hash)

    # load embeddings model

    # - delete chunks for removed files

    # - re-ingest modified files

    # - ingest new files

    # provide a summary of the performed operations



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