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
from pathlib import Path
from llama_index.core.node_parser import (MarkdownNodeParser, SentenceSplitter)

oLogger = logging.getLogger(__name__)


class DocumentParser:

    s_oSTRUCTURED_FORMATS = { ".pdf", ".docx", ".rtf"}  # formats where we bypass the LLamaIndex wrapper to preserve structure and metadata
    s_oHEADER_AWARE_FORMATS = { ".md", ".rst" } # formats handled by LlamaIndex wrapperwith header-aware splitting

    def __init__(self, sFolderPath: str):
        self.sFolderPath = sFolderPath


    def _parseStructured(self, sFilePath: str, bDebugContent: bool = False) -> list[Document]:
        """
        Parse a document by preserving the information about its structure.
        Each structural element (Title, NarrativeText, Table...) becomes
        its own LlamaIndex Document, with category preserved in metadata.
        This parser can be user for formats where structure matters (pdf, docx, rtf)
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


    def parseOneDocument(self, sFilePath: str, bDebugContent: bool = False) -> list[dict]:
        """
        Load and chunk a document file.
        Returns a list of chunked text strings.
        """
        oLogger.debug(f"parseOneDocument. Parsing document: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parseOneDocument. File {sFilePath} does not exist.")
            return []
        
        sFileExtension = oFilePath.suffix.lower()   # the returned exension includes the dot, e.g. ".pdf"

        aoDocuments = []
        if sFileExtension in self.s_oSTRUCTURED_FORMATS:
            oLogger.debug(f"parseOneDocument. Using _parseStructured for file {oFilePath.name} with extension {sFileExtension}")
            aoDocuments = self._parseStructured(sFilePath, bDebugContent)
        else:
            # TODO
            pass

        if not aoDocuments:
            oLogger.warning(f"parseOneDocument. No documents were parsed from file {sFilePath}.")
            return []
    

        if sFileExtension in self.s_oHEADER_AWARE_FORMATS:
            # split on #, ##, ###
            oParser = MarkdownNodeParser()
        else:
            # Used for txt, csv, html AND for pdf/docx/rtf after unstructured
            # (unstructured already gave us element-level granularity,
            #  SentenceSplitter only further splits elements that are too large)
            iChunkSize = 1000       # TODO: make it configurable, what are good numbers?
            iChunkOverlap = 100
            oParser = SentenceSplitter(
                chunk_size=iChunkSize,
                chunk_overlap=iChunkOverlap,
            )

        aoBaseNodes = oParser.get_nodes_from_documents(aoDocuments)
        """
        asChunks = [oNode.get_content() for oNode in aoBaseNodes if oNode.get_content().strip()]

        oLogger.debug(f"parseOneDocument. After splitting, got {len(asChunks)} chunks from file {sFilePath}.")
        """
        aoChunks = [
            {
                "text": oNode.get_content(),
                "metadata": oNode.metadata,  
            }
            for oNode in aoBaseNodes
            if oNode.get_content().strip()
        ]
        return aoChunks
