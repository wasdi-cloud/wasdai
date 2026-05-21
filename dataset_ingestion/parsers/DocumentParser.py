"""
Parser for structured documents: md, rst, txt, csv, pdf.
Returns a list of LlamaIndex Document objects (one per structural element),
preserving metadata such as category (Title, NarrativeText, Table...).
"""

import logging
import re
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
    

    def _splitRstBySections(self, sText: str, sFilePath: str) -> list[Document]:
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

        return aoDocuments


    def parseOneDocument(self, sFilePath: str, bDebugContent: bool = False) -> list[dict]:
        """
        Load and chunk a document file.
        Returns a list of chunked text strings.
        """
        oLogger.info(f"parseOneDocument. Parsing document: {sFilePath}")

        oFilePath = Path(sFilePath)

        if not oFilePath.exists():
            oLogger.warning(f"parseOneDocument. File {sFilePath} does not exist.")
            return []
        
        sFileExtension = oFilePath.suffix.lower()   # the returned exension includes the dot, e.g. ".pdf"

        aoDocuments = []
        if sFileExtension in self.s_oSTRUCTURED_FORMATS:
            oLogger.info(f"parseOneDocument. Using _parseStructured for file {oFilePath.name} with extension {sFileExtension}")
            aoDocuments = self._parseStructured(sFilePath, bDebugContent)
        else:
            oLogger.info(f"parseOneDocument. Using SimpleDirectoryReader for file {oFilePath.name} with extension {sFileExtension}")
            aoDocuments = SimpleDirectoryReader(input_files=[sFilePath]).load_data()

        if not aoDocuments:
            oLogger.warning(f"parseOneDocument. No documents were parsed from file {sFilePath}.")
            return []
    
        if sFileExtension in self.s_oHEADER_AWARE_FORMATS:
            if sFileExtension == ".rst":
                # clean RST syntax first
                sRawText = "\n\n".join([oDoc.get_content() for oDoc in aoDocuments])
                aoDocuments = self._splitRstBySections(sRawText, sFilePath)
                oParser = SentenceSplitter(chunk_size=100, chunk_overlap=100)
            else:
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

        oLogger.info(f"parseOneDocument. After splitting, got {len(asChunks)} chunks from file {sFilePath}.")
        """
        aoChunks = [
            {
                "text": oNode.get_content(),
                "metadata": oNode.metadata,  
            }
            for oNode in aoBaseNodes
            if oNode.get_content().strip()
        ]

        oLogger.info(f"parseOneDocument. Got {len(aoChunks)} chunks from file {sFilePath}.")

        return aoChunks
