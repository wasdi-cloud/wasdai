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


    def __init__(self, asFolderPaths: list[str]):
        self.folderPaths = asFolderPaths


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


    def _scan(self) -> dict[str, str]:
        """
        Scans the folder containing the documents to be processed and computes a hash for each file.
        Returns a dict of {absolute_path: sha256_hash} for all the supported extensions.
        """
        oMergedSnapshot = {}

        for sFolderPath in self.folderPaths:
            oFolderPath = Path(sFolderPath)
            if not oFolderPath.exists() or not oFolderPath.is_dir():
                oLogger.error(f"scan. Invalid folder path: {sFolderPath}")
                raise ValueError(f"Invalid folder path: {sFolderPath}")
            oLogger.info(f"scan. Scanning folder: {sFolderPath}")
            oFolderSnapshot = self._scanOneFolder(oFolderPath)
            oMergedSnapshot.update(oFolderSnapshot)
        return oMergedSnapshot
       
    
    def _scanOneFolder(self, oFolderPath: Path) -> dict[str, str]:
        """
        Scans a single folder and computes a hash for each file.
        Returns a dict of {absolute_path: sha256_hash} for all the supported extensions.
        """

        if not oFolderPath.exists() or not oFolderPath.is_dir():
            oLogger.error(f"_scanOneFolder. Invalid folder path: {oFolderPath}")
            raise ValueError(f"Invalid folder path: {oFolderPath}")
        
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
                    oLogger.warning(f"_scanOneFolder. Error occurred while processing file: {oFilePath}. {oE}")
                    continue

        oLogger.info(f"_scanOneFolder. Scanned {iCount} files. Found {len(oResultDict)} supported files in folder {oFolderPath}")      
        return oResultDict

        
    def findDifference(self, oDbSnapshot: dict[str, str]):
        """
        Compare the current status of the dataset folder against the metadata sotred in the database.
        Produces four lists:
        - new files
        - modified files
        - deleted files
        - unchanged files
        :param oDbSnapshot: dict of {file_path: file_hash} representing the current metadata stored in the database
        """
        oFolderSnapshot = self._scan()
 
        oDatasetFilePaths = set(oFolderSnapshot.keys())
        oDbPaths = set(oDbSnapshot.keys())

        asNewFiles = [oPath for oPath in oDatasetFilePaths - oDbPaths]
        asDeletedFiles = [oPath for oPath in oDbPaths - oDatasetFilePaths]
        asModifiedFiles = [
            oPath for oPath in oDatasetFilePaths & oDbPaths
            if oFolderSnapshot[oPath] != oDbSnapshot[oPath]
        ]
        asUnchangedFiles = [
            oPath for oPath in oDatasetFilePaths & oDbPaths
            if oFolderSnapshot[oPath] == oDbSnapshot[oPath]
        ]

        oLogger.info(f"New files: {len(asNewFiles)}, deleted files: {len(asDeletedFiles)}, modified files: {len(asModifiedFiles)}, unchanged files: {len(asUnchangedFiles)}")
        return oFolderSnapshot, asNewFiles, asDeletedFiles, asModifiedFiles, asUnchangedFiles
