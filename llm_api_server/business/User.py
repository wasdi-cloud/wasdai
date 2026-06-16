from llm_api_server.business.WasdiEntity import WasdiEntity

class User(WasdiEntity):

    def __init__(self, **kwargs):
        self.userId = str()
        self.name = str()
        self.surname = str()
        self.password = str()
        self.validAfterFirstAccess = False
        self.firstAccessUUID = str()
        self.authServiceProvider = str()
        self.googleIdToken = str()
        self.link = str()
        self.description = str()
        self.activeSubscriptionId = str()
        self.activeProjectId = str()
        self.role = str()
        self.type = str()
        self.storageWarningSentDate = float()
        self.publicNickName = str()
        self.skin = str()
        self.defaultNode = str()
        self.lastLogin = str()
        self.confirmationDate = float()
        self.registrationDate = str()

        for key, value in kwargs.items():
            setattr(self, key, value)