import logging
from pathlib import Path

from dataset_ingestion.parsers.BaseParser import BaseParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter

# Import the official tree-sitter core and specific language bindings
from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_java

oLogger = logging.getLogger(__name__)

class CodeParser(BaseParser):
    """Handles files with source code."""

    s_oCODE_FORMATS = {
        ".py":   "python",
        ".java": "java",
    }

    def _get_tree_sitter_parser(self, sLanguage: str) -> Parser:
        """
        Manually construct the Parser using modern tree-sitter (>= 0.22.0) API.
        This bypasses LlamaIndex type-checking bugs by ensuring the exact Parser object is passed.
        """
        if sLanguage == "python":
            oLanguage = Language(tree_sitter_python.language())
        elif sLanguage == "java":
            oLanguage = Language(tree_sitter_java.language())
        else:
            raise ValueError(f"Unsupported language: {sLanguage}")
        
        return Parser(oLanguage)

    def parse(self, sFilePath: str, bDebugContent: bool = False) -> list[dict]:
        """
        Parse a source file using LlamaIndex CodeSplitter.
        Converts the resulting nodes into standard dictionaries using BaseParser.
        """
        oLogger.info(f"parse. Parsing code file: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parse. File {sFilePath} does not exist.")
            return []
        
        sExtension = oFilePath.suffix.lower()
        sProgrammingLanguage = self.s_oCODE_FORMATS.get(sExtension)

        if not sProgrammingLanguage:
            oLogger.warning(f"parse. Unsupported code format: {sExtension}")
            return []

        # load the document
        aoDocuments = SimpleDirectoryReader(input_files=[sFilePath]).load_data()

        if not aoDocuments:
            oLogger.warning(f"parse. No content extracted from {sFilePath}")
            return []

        # inject standard metadata so the chunks inherit it!
        for oDoc in aoDocuments:
            oDoc.metadata["category"] = "Code"

        # Explicitly build the parser to pass into CodeSplitter
        try:
            oTreeSitterParser = self._get_tree_sitter_parser(sProgrammingLanguage)
        except Exception as oE:
            oLogger.error(f"parse. Failed to initialize tree-sitter parser for {sProgrammingLanguage}: {oE}")
            return []

        # Pass the verified parser object directly
        oParser = CodeSplitter(
            language=sProgrammingLanguage,
            parser=oTreeSitterParser,
            chunk_lines=getattr(self, 'chunkSize', 40),       
            chunk_lines_overlap=getattr(self, 'chunkOverlap', 5), 
            max_chars=1500,
        )
        
        aoBaseNodes = oParser.get_nodes_from_documents(aoDocuments)
        aoChunks = self._getChunksFromBaseNodes(aoBaseNodes)

        if bDebugContent:
            for oChunk in aoChunks:
                sText = oChunk.get("text", "")
                oLogger.info(
                    f"--- CODE CHUNK ---\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tfile_name: {oFilePath.name}\n"
                    f"\tPreview:\n\t\t{sText[:500]}\n"
                    f"--- END ---"
                )
        
        oLogger.info(f"parse. After splitting, got {len(aoChunks)} chunks from file {sFilePath}.")
        
        return aoChunks