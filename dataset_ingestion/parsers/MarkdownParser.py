import logging
from pathlib import Path

from dataset_ingestion.parsers.BaseParser import BaseParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

oLogger = logging.getLogger(__name__)

class MarkdownParser(BaseParser):
    """Handles files with Markdown content."""

    def parse(self, sFilePath: str | Path, bDebugContent: bool = False) -> list[dict]:
        """
        Parse a Markdown file using LlamaIndex MarkdownNodeParser.
        Converts the resulting nodes into standard dictionaries using BaseParser.
        """
        # Ensure we are working with a string for logging, and a Path object for operations
        oFilePath = Path(sFilePath)

        oLogger.info(f"parse. Parsing Markdown file: {sFilePath}")

        if not oFilePath.exists():
            oLogger.warning(f"parse. File {sFilePath} does not exist.")
            return []
        
        if oFilePath.suffix.lower() not in [".md", ".markdown"]:
            oLogger.warning(f"parse. File {sFilePath} does not have a supported Markdown extension.")
            return []

        # Load the document
        aoDocuments = SimpleDirectoryReader(input_files=[sFilePath]).load_data()

        if not aoDocuments:
            oLogger.warning(f"parse. No content extracted from {sFilePath}")
            return []

        # Inject standard metadata so the chunks inherit it
        for oDoc in aoDocuments:
            oDoc.metadata["category"] = "Markdown"

        # Initialize the MarkdownNodeParser
        # Note: This parser splits primarily by Markdown headers, preserving structure.
        oParser = MarkdownNodeParser()
        
        # Extract LlamaIndex BaseNodes
        aoBaseNodes = oParser.get_nodes_from_documents(aoDocuments)
        
        # Convert to standard dictionary format using BaseParser's helper
        aoChunks = self._getChunksFromBaseNodes(aoBaseNodes)

        if bDebugContent:
            for oChunk in aoChunks:
                sText = oChunk.get("text", "")
                oLogger.info(
                    f"--- MARKDOWN CHUNK ---\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tfile_name: {oFilePath.name}\n"
                    f"\tPreview:\n\t\t{sText[:500]}\n"
                    f"--- END ---"
                )
        
        oLogger.info(f"parse. After splitting, got {len(aoChunks)} chunks from file {sFilePath}.")
        
        return aoChunks