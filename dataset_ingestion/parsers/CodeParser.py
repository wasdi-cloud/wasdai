import logging
import re

from dataset_ingestion.parsers.BaseParser import BaseParser
from unstructured.partition.auto import partition
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from pathlib import Path

oLogger = logging.getLogger(__name__)

class CodeParser(BaseParser):
    """Handles files with code"""

    s_oCODE_FORMATS = {
        ".py":   "python",
        ".java": "java",
    }

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


    
    def parse(self, sFilePath: str, bDebugContent: bool = False) -> list[Document]:
        """
        Parse a Python source file using LlamaIndex CodeSplitter.
        Splits on logical boundaries (functions, classes) via tree-sitter.
        Each chunk becomes a Document with metadata consistent with other parsers.
        """

        oLogger.info(f"parse. Parsing code file: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parse. File {sFilePath} does not exist.")
            return []
    
        aoDocuments = SimpleDirectoryReader(input_files=[sFilePath]).load_data()

        if not aoDocuments:
            oLogger.warning(f"_parsePython. No content extracted from {sFilePath}")
            return []

        oParser = CodeSplitter(
            language="python",
            chunk_lines=40,           # lines per chunk
            chunk_lines_overlap=5,    # overlap in lines
            max_chars=1500,
        )

        aoNodes = oParser.get_nodes_from_documents(aoDocuments)

        aoResultDocuments = []
        for oNode in aoNodes:
            sText = oNode.get_content().strip()
            if not sText:
                continue

            aoResultDocuments.append(
                Document(
                    text=sText,
                    metadata={
                        "source_path": sFilePath,
                        "category":    "Code",
                        "page_number": None,
                        "file_name":   Path(sFilePath).name,
                    }
                )
            )

            if bDebugContent:
                oLogger.info(
                    f"--- CODE CHUNK ---\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tfile_name: {Path(sFilePath).name}\n"
                    f"\tPreview:\n\t\t{sText[:500]}\n"
                    f"--- END ---"
                )

        return aoResultDocuments