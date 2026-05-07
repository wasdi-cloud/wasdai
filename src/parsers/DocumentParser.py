"""
Parser for structured documents: md, rst, txt, csv, pdf.
Returns a list of LlamaIndex Document objects (one per structural element),
preserving metadata such as category (Title, NarrativeText, Table...).
"""

import logging
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import Document
from unstructured.partition.auto import partition

oLogger = logging.getLogger(__name__)


class DocumentParser:

    def __init__(self, sFolderPath: str):
        self.sFolderPath = sFolderPath


    def parseStructured(self, sFilePath: str, bDebugContent: bool = False) -> list[Document]:
        """
        Parse a document by preserving the information about its structure.
        Each structural element (Title, NarrativeText, Table...) becomes
        its own LlamaIndex Document, with category preserved in metadata.
        """
        oSkipCategories = {"Footer", "Header", "PageBreak", "PageNumber"}
        aoElements = partition(filename=sFilePath)
        aoDocuments = []
        for oEl in aoElements:
            sText = oEl.text.strip() if oEl.text else ""
            if not sText:
                continue
            sCategory = getattr(oEl, "category", "Unknown")
            if sCategory in oSkipCategories:
                continue
            aoDocuments.append(
                Document(
                    text=sText,
                    metadata={
                        "source_path": sFilePath,
                        "category":    sCategory,
                        "page_number": getattr(oEl.metadata, "page_number", None),
                        "file_name":   getattr(oEl.metadata, "filename", None),
                    },
                )
            )
            if bDebugContent:
                oLogger.debug(
                    f"--- DOCUMENT METADATA ---\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tcategory: {sCategory}\n"
                    f"\tpage_number: {getattr(oEl.metadata, 'page_number', None)}\n"
                    f"\tfile_name: {getattr(oEl.metadata, 'filename', None)}\n"
                    f"\tPreview:\n\t\t{sText[:500]}\n"
                    f"--- END ---"
                )
        return aoDocuments


    def parse(self, bDebugContent: bool = False) -> list[Document]:
        oLogger.debug(f"parse. Parsing documents in folder: {self.sFolderPath}")

        # Only use UnstructuredReader for formats where structure matters
        # md is intentionally excluded — LlamaIndex default reader handles it better
        oUnstructuredReader = UnstructuredReader()
        oFileExtractor = {
            ".pdf":  oUnstructuredReader,
            ".docx": oUnstructuredReader,
            ".rtf":  oUnstructuredReader,
        }

        oDirectoryReader = SimpleDirectoryReader(
            self.sFolderPath,
            file_extractor=oFileExtractor,
        )

        aoDocuments = oDirectoryReader.load_data()

        if not aoDocuments:
            oLogger.warning(f"parse. No documents found in folder: {self.sFolderPath}")
            return []

        if bDebugContent:
            for oDocument in aoDocuments:
                sMetadataDisplay = "\n".join(
                    [f"\t\t{k}: {v}" for k, v in oDocument.metadata.items()]
                )
                oLogger.debug(
                    f"--- DOCUMENT METADATA ---\n"
                    f"\tFile: {oDocument.metadata.get('file_name')}\n"
                    f"\tCategory: {oDocument.metadata.get('category', 'Unknown')}\n"
                    f"\tMetadata:\n{sMetadataDisplay}\n"
                    f"\tPreview:\n\t\t{oDocument.text[:500].strip()}\n"
                    f"--- END ---"
                )

        return aoDocuments  # return the full list, structure intact