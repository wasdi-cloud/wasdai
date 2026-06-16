from llm_api_server.business.User import User
from llm_api_server.data.MongoRepository import MongoRepository

class UserRepository(MongoRepository):

    def __init__(self):
        super().__init__()
        self.m_sCollectionName = "users"
        self.m_sEntityClassName = f"{User.__module__}.{User.__qualname__}"