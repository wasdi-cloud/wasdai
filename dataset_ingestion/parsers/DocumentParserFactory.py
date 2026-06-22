import logging

from .StructuredParser import StructuredParser
from .RSTParser import RSTParser
from .BaseParser import BaseParser
from .CodeParser import CodeParser
from .MarkdownParser import MarkdownParser

from pathlib import Path

olOgger = logging.getLogger(__name__)

class DocumentParserFactory:
    """Creates the appropriate parser based on file extension."""
    
    # Map extensions to their respective parser classes
    _s_oPARSER_MAP = {
        
        ".pdf": [StructuredParser, 1000, 100],
        ".docx": [StructuredParser, 1000, 100],  # TODO: check
        ".rtf": [StructuredParser, 1000, 100], # TODO: check
        ".rst": [RSTParser, 500, 100],
        ".md": [MarkdownParser, 1000, 100],
        ".markdown": [MarkdownParser, 1000, 100],
        # for code, the parameters are not the chunk size and overlap, but the lines and line overlap, which will be mapped inside the CodeParser
        ".py":   [CodeParser, 40, 5],            
        ".java": [CodeParser, 40, 5]
    }

    @classmethod
    def getParser(cls, sFilePath: str) -> BaseParser:
        oPath = Path(sFilePath)
        sExtension = oPath.suffix.lower()
        
        # Look up the correct parser class, default to DefaultParser if not found
        aoParserInfo = cls._s_oPARSER_MAP.get(sExtension)
        if not aoParserInfo:
            olOgger.error(f"DocumentParserFactory.getParser. No parser found for extension: {sExtension}")
            return None 
        else:
            oParserClass = aoParserInfo[0]
            iChunkSize = aoParserInfo[1]
            iChunkOverlap = aoParserInfo[2]
            return oParserClass(iChunkSize, iChunkOverlap)