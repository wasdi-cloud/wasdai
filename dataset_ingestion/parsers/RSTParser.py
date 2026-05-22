import logging
import re

from dataset_ingestion.parsers.BaseParser import BaseParser
from unstructured.partition.auto import partition
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

oLogger = logging.getLogger(__name__)

class RSTParser(BaseParser):
    """Handles RST files"""

    def parse(self, sFilePath: str, bDebugContent: bool = False) -> list[dict]:
        oLogger.info(f"parse. Parsing RST file: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parse. File {sFilePath} does not exist.")
            return []
        
        aoDocuments = SimpleDirectoryReader(input_files=[sFilePath]).load_data()

        if not aoDocuments:
            oLogger.warning(f"parse. No sections were parsed from file {sFilePath}.")
            return []
        
        # clean RST syntax first
        sRawText = "\n\n".join([oDoc.get_content() for oDoc in aoDocuments])
        aoDocuments = self._splitRstBySections(sRawText, sFilePath, bDebugContent)

        oParser = SentenceSplitter(
            chunk_size=self.chunkSize,
            chunk_overlap=self.chunkOverlap,
        )

        aoBaseNodes = oParser.get_nodes_from_documents(aoDocuments)
        aoChunks = self._getChunksFromBaseNodes(aoBaseNodes)
    
        oLogger.info(f"parse. After splitting, got {len(aoChunks)} chunks from file {sFilePath}.")

        return aoChunks



    def _splitRstBySections(self, sText: str, sFilePath: str, bDebugContent: bool = False) -> list[Document]:
        """
        Split RST text into sections based on header underlines.
        Each section becomes a Document with metadata consistent
        with the PDF parser output.
        """
        sText = sText.replace('\r\n', '\n').replace('\r', '\n')

        oHeaderPattern = re.compile(
            r'^(.+)\n([=\-~\^"\'`#\*\+]{3,})\s*$',
            flags=re.MULTILINE
        )

        aoSections = []
        iLastEnd = 0
        sCurrentTitle = Path(sFilePath).stem  # filename as fallback for preamble

        for oMatch in oHeaderPattern.finditer(sText):
            sSectionText = sText[iLastEnd:oMatch.start()].strip()
            if sSectionText:
                aoSections.append((sCurrentTitle, sSectionText))
            sCurrentTitle = oMatch.group(1).strip()
            iLastEnd = oMatch.end()

        sSectionText = sText[iLastEnd:].strip()
        if sSectionText:
            aoSections.append((sCurrentTitle, sSectionText))

        aoDocuments = []
        for sTitle, sBody in aoSections:
            sCleanedBody = self._cleanRstSyntax(sBody)
            if not sCleanedBody.strip():
                continue
            aoDocuments.append(
                Document(
                    text=sCleanedBody,
                    metadata={
                        "source_path": sFilePath,
                        "category":    "Section",
                        "page_number": None,          # not applicable for RST
                        "file_name":   Path(sFilePath).name,
                        "section_title": sTitle,      # RST-specific, extra field
                    }
                )
            )

            if bDebugContent:
                oLogger.info(
                    f"--- DOCUMENT METADATA --\n"
                    f"\tsource_path: {sFilePath}\n"
                    f"\tfile_name: {Path(sFilePath).name}\n"
                    f"\tsection_title: {sTitle}\n"
                    f"\tPreview:\n\t\t{sCleanedBody[:500]}\n"
                    f"--- END ---"
                )

        return aoDocuments
    

    def _cleanRstSyntax(self, sText: str) -> str:
        """
        Clean RST syntax from the given text string.
        """
        # Normalise Windows line endings first
        sCleaned = sText.replace('\r\n', '\n').replace('\r', '\n')

        # Remove Directives and Comments (lines starting with '.. ')
        sCleaned = re.sub(r'^\s*\.\.\s+.*$', '', sCleaned, flags=re.MULTILINE)

        # Remove Header Underlines/Overlines (lines of ===, ---, ~~~, etc.)
        sCleaned = re.sub(r'^[=\-\`:\.\'\"\~\^\_\*\+#]{3,}$', '', sCleaned, flags=re.MULTILINE)

        # Remove Markdown Headers (e.g., # Header, ## Subheader)
        sCleaned = re.sub(r'^#+\s+', '', sCleaned, flags=re.MULTILINE)

        # Clean up Links and Hyperlinks
        # `Link Text <http://example.com>`_
        sCleaned = re.sub(r'`([^<]+)\s*<[^>]+>`_+', r'\1', sCleaned)
        # Anonymous links: text__ or text_
        sCleaned = re.sub(r'(\w+)(__?)(?=\s|$)', r'\1', sCleaned)

        # Strip Inline Formatting (Bold, Italic, Inline Code)
        # Removes **, *, __, _, `, and `` without removing the words inside them
        sCleaned = re.sub(r'(\*\*|__|\*|_|``|`)', '', sCleaned)

        # Clean up Roles/Interpreted Text (e.g., :func:`open`, :ref:`label`)
        sCleaned = re.sub(r':\w+:`([^`]+)`', r'\1', sCleaned)

        # Normalize whitespace
        sCleaned = re.sub(r'\n{3,}', '\n\n', sCleaned)
        sCleaned = sCleaned.strip()

        return sCleaned