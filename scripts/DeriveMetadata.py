"""
Derive the new fields for the metadata (source_type, audience, programming_language, component)
starting from the metadata fields already existing (category, sourcePath)
"""

import logging
import json
from dataset_ingestion.ChromaStore import ChromaStore
from pathlib import Path
from typing import Any
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
oLogger = logging.getLogger(__name__)

BATCH_SIZE = 500

def deriveMetadata(oExistingMetadata: dict[str, Any]) -> dict[str, Any]:
    """
    Get the metadata of a chun and return ONLY the new fields to add. 
    """
    sCategory = oExistingMetadata.get("category", "")
    sSourcePath = oExistingMetadata.get("sourcePath", "")
    sExtension = Path(sSourcePath).suffix.lower()

    oNewFields: dict[str, Any] = {}

    if sExtension == ".java":
        oNewFields["source_type"] = "codebase"
        oNewFields["audience"] = "dev"
        oNewFields["language"] = "java"
        oNewFields["component"] = "platform"

    elif sExtension == ".py":
        oNewFields["source_type"] = "codebase"
        oNewFields["audience"] = "dev"
        oNewFields["language"] = "python"
        oNewFields["component"] = "eo_app"

    elif sExtension == ".md":
        oNewFields["source_type"] = "user_doc"
        oNewFields["component"] = "eo_app"
        oNewFields["audience"] = "any"

    elif sExtension == ".rst":
        oNewFields["source_type"] = "user_doc"
        oNewFields["component"] = "platform"
        oNewFields["audience"] = "any"

    elif not sExtension:
        oNewFields["source_type"] = "unknown"
        oNewFields["component"] = "unknown"
        oNewFields["audience"] = "any"

    else:
        # future extensions
        oNewFields["source_type"] = "user_doc"
        oNewFields["component"] = "unknown"
        oNewFields["audience"] = "any"

    if sCategory == "Code" and oNewFields.get("source_type") != "codebase":
        oNewFields["_warning"] = f"category=Code ma estensione={sExtension} non riconosciuta"


    return oNewFields

def updateMetadataInStore(oChromaStore: ChromaStore, bDryRun: bool = True):
    oResult = oChromaStore.collection.get(include=["metadatas"])
    asIds = oResult["ids"]
    aoMetadatas = oResult["metadatas"]

    oLogger.info(f"Found {len(asIds)} chunk in the collection")

    asIdsToUpdate = []
    aoMetadatasToUpdate = []
    oCategoryCount: dict[str, int] = {}
    oLangComponentCount: dict[str, int] = {}

    for sId, oMetadata in zip(asIds, aoMetadatas):
        oNewFields = deriveMetadata(oMetadata)
        oMergedMeta = {**oMetadata, **oNewFields}
        sCategory = oMetadata.get("category", "UNKNOWN")
        oCategoryCount[sCategory] = oCategoryCount.get(sCategory, 0) + 1
        sKey = f"{oNewFields.get('language', '-')}/{oNewFields.get('component', '-')}"
        oLangComponentCount[sKey] = oLangComponentCount.get(sKey, 0) + 1
        asIdsToUpdate.append(sId)
        aoMetadatasToUpdate.append(oMergedMeta)

    oLogger.info(f"Summary for original category {oCategoryCount}")
    oLogger.info(f"Summary for derivative language/component {oLangComponentCount}")

    if bDryRun:
        oLogger.info("DRY RUN - nothing updated")
        oLogger.info("Example (top 5)")
        for sId, oMeta in list(zip(asIdsToUpdate, aoMetadatasToUpdate))[:5]:
            oLogger.info(f"\n  id: {sId}")
            oLogger.info(f"  metadata: {oMeta}")

        aoUnknowns = [
            (sId, oMeta) for sId, oMeta in zip(asIdsToUpdate, aoMetadatasToUpdate)
            if oMeta.get("component") == "unknown"
        ]
        logging.info(f"\n--- {len(aoUnknowns)} chunks with component=unknown ---")
        oPathSample = {}
        for sId, oMeta in aoUnknowns:
            sPath = oMeta.get("sourcePath", "MISSING")
            print("**** ")
            sExt = sPath.rsplit(".", 1)[-1] if "." in sPath else "NO_EXTENSION"
            oPathSample[sExt] = oPathSample.get(sExt, 0) + 1

        for sExt, iCount in sorted(oPathSample.items(), key=lambda x: -x[1]):
            print(f"  {iCount}x  estensione: .{sExt}")
        return

    for i in range(0, len(asIdsToUpdate), BATCH_SIZE):
        asBatchIds = asIdsToUpdate[i:i + BATCH_SIZE]
        aoBatchMetas = aoMetadatasToUpdate[i:i + BATCH_SIZE]
        oChromaStore.collection.update(
            ids=asBatchIds,
            metadatas=aoBatchMetas
        )
        oLogger.info(f"Aggiornati {i + len(asBatchIds)}/{len(asIdsToUpdate)} chunk")

    oLogger.info("\nUpdate completed")


def deletePdfChunksById(oChromaStore: ChromaStore, bDryRun: bool = True):
    oResult = oChromaStore.collection.get(include=["metadatas"])
    asIds = oResult["ids"]
    aoMetadatas = oResult["metadatas"]

    asIdsToDelete = [
        sId for sId, oMeta in zip(asIds, aoMetadatas)
        if (oMeta.get("sourcePath") or "").lower().endswith(".pdf")
    ]

    print(f"Found {len(asIdsToDelete)} chunks from pdf files to delete")

    if bDryRun:
        logging.info("--- DRY RUN: no deletion being performed ---")
        return

    oChromaStore.collection.delete(ids=asIdsToDelete)
    print(f"Deleted: {len(asIdsToDelete)} chunk.")

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

if __name__ == "__main__":
    sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
    oConfig = readConfigFile(sConfigFilePath)

    oChromaStore = ChromaStore(
        sPersistDirectory=oConfig.chromaStore.persistDirectory,
        sCollectionName=oConfig.chromaStore.collectionName
    )

    # deletePdfChunksById(oChromaStore, bDryRun=False)
    # dry run
    #updateMetadataInStore(oChromaStore, bDryRun=True)

    # actual update
    #updateMetadataInStore(oChromaStore, bDryRun=False)

    # check up to see that the update ran succesfully
    oCheck = oChromaStore.collection.get(limit=3, include=["metadatas"])
    for oMetadata in oCheck["metadatas"]:
        print(oMetadata)