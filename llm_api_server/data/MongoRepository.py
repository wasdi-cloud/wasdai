import logging

from llm_api_server.MongoDBClient import MongoDBClient
from utils import Utils

class MongoRepository:
    # name of the database connected to this repository
    s_sDB_NAME = "wasdi"

    def __init__(self):
        self.m_sCollectionName = None
        self.m_sId = None
        self.m_sEntityClassName = None

    def getCollection(self):
        """
        Retrieves from the database a collection
        :return: the collection if present, None otherwise
        """
        oCollection = None
        try:
            oMongoClient = MongoDBClient()
            oDatabase = oMongoClient.client[MongoRepository.s_sDB_NAME]

            if oDatabase is None:
                logging.warning(f"MongoRepository.getCollection. Database named '{MongoRepository.s_sDB_NAME}' not found in Mongo")
                return None

            oCollection = oDatabase[self.m_sCollectionName]
        except Exception as oEx:
            logging.error(f"MongoRepository.getCollection. Exception retrieving the collection {oEx}")

        return oCollection


    def getEntityById(self, sEntityId):
        """
        Given the id of an entity, retrieves it from the database
        :param sEntityId: the id of the entity (not the Mongo _id, but the RISE internal id of the entity)
        :return: the entity with the required id, None otherwise
        """
        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.findEntityById. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return None

        
            oRetrievedResult = oCollection.find({self.m_sId: sEntityId})

            if oRetrievedResult is None:
                logging.info(f"MongoRepository.findEntityById. No results retrieved from db")
                return None

            aoRetrievedEntities = []
            for oResEntity in oRetrievedResult:
                oEntityClass = Utils.getClass(self.m_sEntityClassName)
                aoRetrievedEntities.append(oEntityClass(**oResEntity))

            if len(aoRetrievedEntities) > 0:
                return aoRetrievedEntities[0]
            else:
                return None
        except Exception as oEx:
            logging.error(f"MongoRepository.findEntityById. Exception {oEx}")

        return None

    def listAllEntities(self):
        """
        List all the entities in a collection
        :return: the full list of entities in a collection
        """
        oCollection = self.getCollection()

        if oCollection is None:
            logging.warning(f"MongoRepository.listAllEntities. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
            return None

        try:
            oRetrievedResult = oCollection.find({})

            if oRetrievedResult is None:
                logging.info(f"MongoRepository.listAllEntities. No results retrieved from db")
                return None

            aoRetrievedEntities = []
            for oResEntity in oRetrievedResult:
                oEntityClass = Utils.getClass(self.m_sEntityClassName)
                aoRetrievedEntities.append(oEntityClass(**oResEntity))

            return aoRetrievedEntities

        except Exception as oEx:
            logging.error(f"MongoRepository.listAllEntities. Exception {oEx}")

        return None

    def getAllEntitiesById(self, asEntityIds):
        """
        Given a list of entities' ids, retrieves from a collection the list of entities matching those ids
        :param asEntityIds: list of entities' ids to retrieve (not the Mongo _id, but the RISE internal id of the entity)
        :return: the list of entities matching the ids passed as parameters
        """
        try:
            if asEntityIds is None or len(asEntityIds) == 0:
                logging.warning("MongoRepository.findAllEntitiesById. No ids specified")
                return None

            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.findAllEntitiesById. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return None

            oRetrievedResult = oCollection.find({self.m_sId: {"$in": asEntityIds}})

            if oRetrievedResult is None:
                logging.debug(f"MongoRepository.findAllEntitiesById. No results retrieved from db")
                return None

            aoRetrievedEntities = []
            for oResMap in oRetrievedResult:
                oEntityClass = Utils.getClass(self.m_sEntityClassName)
                aoRetrievedEntities.append(oEntityClass(**oResMap))

            return aoRetrievedEntities

        except Exception as oEx:
            logging.error(f"MongoRepository.findAllEntitiesById. Exception {oEx}")

        return None

    def getEntitiesByField(self, aoAttributeMap):
        """
        Given a dictionary, returns the list of the entities matching all the key-value pairs in the dictionary
        :param aoAttributeMap: a dictionary of all the key-value pairs that the retrieved entities should match
        :return: the list of entities matching the ket-value pairs in the dictionary
        """

        if aoAttributeMap is None or aoAttributeMap.items() == 0:
            return None

        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.getEntitiesByField. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return None

            oRetrievedResult = oCollection.find(aoAttributeMap)

            if oRetrievedResult is None:
                logging.info(f"MongoRepository.getEntitiesByField. No results retrieved from db")
                return None

            aoRetrievedEntities = []
            for oResEntity in oRetrievedResult:
                oEntityClass = Utils.getClass(self.m_sEntityClassName)
                aoRetrievedEntities.append(oEntityClass(**oResEntity))

            return aoRetrievedEntities

        except Exception as oEx:
            logging.error(f"MongoRepository.getEntitiesByField. Exception {oEx}")

        return None

    def addEntity(self, oEntity):
        """
        Insert an entity in a collection
        :param oEntity: the entity to add to the collection
        :return: True if the entity was successfully added to the collection, False otherwise
        """
        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.addEntity. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return False

            oCollection.insert_one(vars(oEntity))

            return True
        except Exception as oEx:
            logging.error(f"MongoRepository.addEntity. Exception {oEx}")

        return False

    def updateEntity(self, oEntity):
        """
        Given an entity, updates the entry with the same id in the database
        :param oEntity: the entity to update
        :return: True if the update was successful, False otherwise
        """
        if oEntity is None or self.m_sId not in vars(oEntity):
            logging.warning("MongoRepository.updateEntity. The provided entity is None or is missing the 'id' filed")
            return False

        sEntityId = getattr(oEntity, self.m_sId)
        oQuery = {self.m_sId: sEntityId}
        oUpdatedDocument = {"$set": vars(oEntity)}

        try:
            if "_id" in oUpdatedDocument["$set"]:
                del oUpdatedDocument["$set"]["_id"]
        except Exception as oEx:
            logging.error(f"MongoRepository.updateEntity. Exception removing _id from updated document: {oEx}")

        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.updateEntity. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return False

            oResult = oCollection.update_one(oQuery, oUpdatedDocument)

            if oResult.modified_count > 0:
                return True

            logging.warning("MongoRepository.updateEntity. No document updated in the database")

        except Exception as oEx:
            logging.error(f"MongoRepository.updateEntity. Exception {oEx}")

        return False


    def updateAllEntities(self, aoEntities):
        """
        Given a list of entities, updates them in the collection, based on their 'id' field
        :param aoEntities: the list of entities to update
        :return: the number of updated entities
        """
        iUpdatedEntities = 0

        if aoEntities is None or len(aoEntities) < 1:
            logging.warning("MongoRepository.updateAllEntities. The provided list of entities is None or empty")
            return iUpdatedEntities

        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(
                    f"MongoRepository.updateAllEntities. Collection {self.m_sCollectionName} not "
                    f"found in {MongoRepository.s_sDB_NAME} database")
                return iUpdatedEntities

            for oEntity in aoEntities:

                try: 
                    if not hasattr(oEntity, self.m_sId):
                        logging.warning(f"MongoRepository.updateAllEntities. Entity missing 'id' {oEntity}")
                        continue
                    sEntityId = getattr(oEntity, self.m_sId)
                    oQuery = {self.m_sId: sEntityId}
                    oUpdatedDocument = {"$set": vars(oEntity)}

                    try:
                        if "_id" in oUpdatedDocument["$set"]:
                            del oUpdatedDocument["$set"]["_id"]
                    except Exception as oEx:
                        logging.error(f"MongoRepository.updateAllEntities. Exception removing _id from updated document: {oEx}")

                    oResult = oCollection.update_many(oQuery, oUpdatedDocument)

                    if oResult.modified_count > 0:
                        iUpdatedEntities += 1
                    else:
                        logging.warning(f"MongoRepository.updateAllEntities. Entity {sEntityId} not updated")
            
                except Exception as oEx2:
                    logging.error(f"MongoRepository.updateAllEntities. Exception while processing one entity: {oEx2}")
        except Exception as oEx:
            logging.error(f"MongoRepository.updateAllEntities. Exception {oEx}")

        return iUpdatedEntities


    def deleteEntity(self, sEntityId):
        """
        Given an entity id, delete the corresponding entry in the database
        :param sEntityId: the id of the entity to delete
        :return: True if the deletion was successful, False otherwise
        """
        if sEntityId is None or sEntityId == '':
            logging.warning("MongoRepository.deleteEntity. The provided entity is None or empty")
            return False

        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.deleteEntity. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return False

            oResult = oCollection.delete_one({self.m_sId: sEntityId})

            if oResult.deleted_count > 0:
                return True

            logging.warning("MongoRepository.deleteEntity. No entity deleted from the database")

        except Exception as oEx:
            logging.error(f"MongoRepository.deleteEntity. Exception {oEx}")

        return False


    def deleteAllEntitesById(self, asEntityIds):
        """
        Given a list of entity ids, delete the corresponding entries in the database
        :param asEntityIds: the list of ids of the entity to delete
        :return: True if the deletion was successful, False otherwise
        """
        if asEntityIds is None or len(asEntityIds) == 0:
            logging.warning("MongoRepository.deleteAllEntitiesById. The provided entity is None or empty")
            return False

        try:
            oCollection = self.getCollection()

            if oCollection is None:
                logging.warning(f"MongoRepository.deleteAllEntitiesById. Collection {self.m_sCollectionName} not found in {MongoRepository.s_sDB_NAME} database")
                return False

            oResult = oCollection.delete_many({self.m_sId: {"$in": asEntityIds}})

            if oResult.deleted_count > 0:
                return True

            logging.warning("MongoRepository.deleteAllEntitiesById. No entity deleted from the database")

        except Exception as oEx:
            logging.error(f"MongoRepository.deleteAllEntitiesById. Exception {oEx}")

        return False