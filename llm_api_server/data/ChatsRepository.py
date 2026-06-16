from llm_api_server.business.Chat import Chat
from llm_api_server.data.MongoRepository import MongoRepository

class ChatRepository(MongoRepository):

    def __init__(self):
        super().__init__()
        self.m_sCollectionName = "chats"
        self.m_sEntityClassName = f"{Chat.__module__}.{Chat.__qualname__}"