import json
import logging
from types import SimpleNamespace

from pymongo import MongoClient


class MongoDBClient:
    """
    Class dedicated to the creation of a single instance of the Mongo DB client
    """

    _s_oConfig = None
    _s_oInstance = None

    def __new__(cls):
        if cls._s_oInstance is None:
            cls._s_oInstance = super(MongoDBClient, cls).__new__(cls)
            sConnectionString = cls._getConnectionString()
            try:
                cls._s_oInstance.client = MongoClient(sConnectionString)
            except Exception as oEx:
                logging.error("MongoDBClient.__new__: exception " + str(oEx))

        return cls._s_oInstance


    @staticmethod
    def _getConnectionString():
        if MongoDBClient._s_oConfig is not None:
            sConnectionString = "mongodb://" + MongoDBClient._s_oConfig.mongoMain.user + ":" + MongoDBClient._s_oConfig.mongoMain.password
            sConnectionString = sConnectionString + "@" + MongoDBClient._s_oConfig.mongoMain.address
            sConnectionString = sConnectionString + "/?authSource=" + MongoDBClient._s_oConfig.mongoMain.dbName
            return sConnectionString

        logging.warning("MongoDBClient._getConnectionString. Configuration is None. Using default connection string")
        return "mongodb://localhost:27017"