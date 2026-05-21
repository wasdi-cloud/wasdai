from StructuredParser import StructuredParser
from RSTParser import RSTParser
from BaseParser import BaseParser

from pathlib import Path

class DocumentParserFactory:
    """Creates the appropriate parser based on file extension."""
    
    # Map extensions to their respective parser classes
    _s_oPARSER_MAP = {
        ".pdf": StructuredParser,
        ".docx": StructuredParser,  # TODO: check
        ".rtf": StructuredParser, # TODO: check
        ".rst": RSTParser
    }

    @classmethod
    def getParser(cls, sFilePath: str) -> BaseParser:
        oPath = Path(sFilePath)
        sExtension = oPath.suffix.lower()
        
        # Look up the correct parser class, default to DefaultParser if not found
        oParserClass = cls._s_oPARSER_MAP.get(sExtension)
        if not oParserClass:
            raise ValueError(f"DocumentParserFactory.getParser. No parser found for extension: {sExtension}")
        return oParserClass()