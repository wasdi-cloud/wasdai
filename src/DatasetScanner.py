"""
Scans the folder containing the documents to be processed and computes a hash for each file.
Returns a dict of {file_path: file_hash} for all the supported extensions.
"""

import os
import logging
import hashlib
from pathlib import Path

oLogger = logging.getLogger(__name__)

class DatasetScanner:

    s_asSUPPORTED_EXTENSIONS = {
        # Documents
        ".md", ".rst", ".txt", ".csv", ".html", ".pdf",
        # Code
        ".py", ".java",
    }


    def __init__(self, sFolderPath: str):
        self.folderPath = sFolderPath


    def _computeFileHash(self, sFilePath: str) -> str:
        """
        Computes the sha256 hash for the given file.
        Returns the hash as a string.
        """
        oH = hashlib.sha256()   # empty hash calculator
        with open(sFilePath, "rb") as oFile:
            # read the file in chunks to avoid memory issues with large files
            for yChunk in iter(lambda: oFile.read(8192), b""):  # 8KB  chunks
                oH.update(yChunk)
        return oH.hexdigest()


    def scan(self) -> dict[str, str]:
        """
        Scans the folder containing the documents to be processed and computes a hash for each file.
        Returns a dict of {absolute_path: sha256_hash} for all the supported extensions.
        """
        oFolderPath = Path(self.folderPath)

        if not oFolderPath.exists() or not oFolderPath.is_dir():
            oLogger.error(f"scan. Invalid folder path: {self.folderPath}")
            raise ValueError(f"Invalid folder path: {self.folderPath}")
        
        oLogger.debug(f"scan. Scanning folder: {self.folderPath}")

        oResultDict = {}
        iCount = 0

        for sRootPath, _, asFileNames in os.walk(oFolderPath):
            for sFileName in asFileNames:
                iCount += 1
                oFilePath = Path(sRootPath) / sFileName
                sFileExtension = oFilePath.suffix.lower()
                if sFileExtension not in self.s_asSUPPORTED_EXTENSIONS:
                    continue
                try:
                    oResultDict[str(oFilePath)] = self._computeFileHash(str(oFilePath))
                except Exception as oE:
                    oLogger.warning(f"scan. Error occurred while processing file: {oFilePath}. {oE}")
                    continue

        oLogger.info(f"scan. Scanned {iCount} files. Found {len(oResultDict)} supported files in folder {self.folderPath}")      
        return oResultDict