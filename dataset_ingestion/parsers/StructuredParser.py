import logging

from dataset_ingestion.parsers.BaseParser import BaseParser
from unstructured.partition.auto import partition
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

oLogger = logging.getLogger(__name__)

class StructuredParser(BaseParser):
    """Handles PDF"""

    def parse(self, sFilePath: str, bDebugContent: bool = False) -> list[dict]:
        oLogger.info(f"parse. Parsing structured file: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parse. File {sFilePath} does not exist.")
            return []
        
        aoDocuments = self._getDocumentSections(sFilePath, bDebugContent)

        if not aoDocuments:
            oLogger.warning(f"parse. No sections were parsed from file {sFilePath}.")
            return []
        
        oParser = SentenceSplitter(
            chunk_size=self.chunkSize,
            chunk_overlap=self.chunkOverlap,
        )

        aoBaseNodes = oParser.get_nodes_from_documents(aoDocuments)
        aoChunks = self._getChunksFromBaseNodes(aoBaseNodes)
    
        oLogger.info(f"parse. After splitting, got {len(aoChunks)} chunks from file {sFilePath}.")

        return aoChunks


    def _getDocumentSections(self, sFilePath: str, bDebugContent: bool = False) -> list[Document]:
        """
        Parse a document by preserving the information about its structure.
        Each structural element (Title, NarrativeText, Table...) becomes
        its own LlamaIndex Document, with category preserved in metadata.
        This parser can be user for formats where structure matters (pdf, docx, rtf)
        """
        oSkipCategories = {"Footer", "Header", "PageBreak", "PageNumber", "Unknown"}
        aoElements = partition(filename=sFilePath)
        aoDocuments = []
        for oEl in aoElements:
            sText = oEl.text.strip() if oEl.text else ""
            if not sText:
                continue
            sCategory = getattr(oEl, "category", "Unknown")
            if sCategory in oSkipCategories:
                if bDebugContent:
                    oLogger.info(f"_getDocumentSections. Skipping category {sCategory} for content {sText[:100]}")
                continue
            aoDocuments.append(
                Document(
                    text = sText,
                    metadata = {
                        "source_path": sFilePath,
                        "category":    sCategory,
                        "page_number": getattr(oEl.metadata, "page_number", None),
                        "file_name":   getattr(oEl.metadata, "filename", None),
                    },
                )
            )
            if bDebugContent:
                oLogger.info(
                    f"--- DOCUMENT METADATA ---\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tcategory: {sCategory}\n"
                    f"\tpage_number: {getattr(oEl.metadata, 'page_number', None)}\n"
                    f"\tfile_name: {getattr(oEl.metadata, 'filename', None)}\n"
                    f"\tPreview:\n\t\t{sText[:500]}\n"
                    f"--- END ---"
                )
        return aoDocuments