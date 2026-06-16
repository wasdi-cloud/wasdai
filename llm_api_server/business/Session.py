from llm_api_server.business.WasdiEntity import WasdiEntity

class Session(WasdiEntity):

    def __init__(self, **kwargs):
        self.sessionId = str()
        self.userId = str()
        self.loginDate = float()
        self.lastTouch = float()

        for key, value in kwargs.items():
            setattr(self, key, value)