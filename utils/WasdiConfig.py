import json
from types import SimpleNamespace

class WasdiConfig:

    _oInstance = None

    def __new__(cls, *args, **kwargs):
        # Create the instance only if it doesn't exist yet
        if cls._oInstance is None:
            cls._oInstance = super(WasdiConfig, cls).__new__(cls)
            cls._oInstance._initialized = False
        return cls._oInstance

    def __init__(self, sConfigFilePath=None):
        # skip if the file has already been loaded
        if self._initialized:
            return
        
        if not sConfigFilePath:
            raise ValueError("WasdiConfig.__init__: config file path needed for the initial load.")
        
        with open(sConfigFilePath, "r") as oConfigFile:
            sConfigContent = oConfigFile.read()
        
        oConfig = json.loads(sConfigContent, object_hook=lambda d: SimpleNamespace(**d))
        
        # transfer the parsed data onto this instance's dictionary
        self.__dict__.update(oConfig.__dict__)
        self.myFilePath = sConfigFilePath
        
        # mark as initialized so we don't read the file again
        self._initialized = True