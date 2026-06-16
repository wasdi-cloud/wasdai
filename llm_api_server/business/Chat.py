from llm_api_server.business.WasdiEntity import WasdiEntity

class Chat(WasdiEntity):

    def __init__(self, **kwargs):
        self.chatId = str()
        self.userId = str()
        self.startDate = float()
        self.prompts = list()
        self.answers = list()

        for key, value in kwargs.items():
            setattr(self, key, value)