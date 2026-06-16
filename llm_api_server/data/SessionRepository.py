from llm_api_server.business.Session import Session
from llm_api_server.data.MongoRepository import MongoRepository

class SessionRepository(MongoRepository):

    def __init__(self):
        super().__init__()
        self.m_sCollectionName = "sessions"
        self.m_sEntityClassName = f"{Session.__module__}.{Session.__qualname__}"

