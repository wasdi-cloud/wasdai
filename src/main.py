import logging
from utils.LoggingConfiguration import setupLogging

setupLogging()

from parsers.DocumentParser import DocumentParser

oLogger = logging.getLogger(__name__)


def main(sFolderPath: str):
    oDocumentParser = DocumentParser(sFolderPath)
    sContent = oDocumentParser.parseStructured(
        sFilePath="C:\\WASDI\\GIT\\wasdai\\test_dataset\\Tutorial_eDrift_v05.docx.pdf", 
        bDebugContent=True)
    # oLogger.info(f"main. Parsed content:\n{sContent}")
    # oLogger.info(f"main. Parsed content:\n{sContent}")

if __name__ == "__main__":
    sFolderPath = "test_dataset"    # TODO: from where should I read this path? from env variable? from command line argument? 
    main(sFolderPath)